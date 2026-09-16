#![allow(unused)]
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, watch};

use opencv::core::MatTraitConst;
use opencv::prelude::*;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Vector},
    imgcodecs,
};

use crate::alpr::GateEvent;
use crate::config::AppConfig;
use crate::frame_src::FrameSource;
use crate::inference::{CountCarsIntelligence, TrackedObject};


//************************
//*
//* Main Run Camera Thread
//*
// loop:
//     1. Receive configuration updates
//     2. Compute FPS and write it to the frame
//     3. Receive frame from the video source
//     4. Check if should_run_inference:
//         a. If yes, run YOLO inference
//         b. for every object larger than the minimum box area, draw a bounding box.
//     5. Send frame for streaming and archiving
//
//*************************
pub fn run_camera_loop<A: CountCarsIntelligence, F: FrameSource>(
    mut rx_config: watch::Receiver<AppConfig>,
    gate_tx_cam1: broadcast::Sender<GateEvent>,
    stream_tx_cam1: broadcast::Sender<Vec<u8>>,
    mut vision_system: A,
    mut video_source: F,
) {
    let mut is_alpr_active = false;
    let mut last_alpr_trigger = Instant::now() - Duration::from_secs(3);

    println!("Starting High-Speed Camera/YOLO Loop...");

    let mut frame_count = 0;
    let mut last_fps_time = Instant::now();
    let mut current_fps = 0.0;

    loop {
        let mut frame = match video_source.next_frame() {
            Some(f) => f,
            None => {
                println!("Camera 1: Video source ended, restarting...");
                if video_source.restart() {
                    continue;
                } else {
                    break;
                }
            }
        };

        // 0. Increment frame count to determine later if inference should be run.
        frame_count += 1;

        // A. Check for live configuration updates with zero lag
        if rx_config.has_changed().unwrap_or(false) {
            let new_config = rx_config.borrow_and_update().clone();
            println!(
                "Camera 1: Camera Thread received new config: {:?}",
                new_config
            );
        }
        let current_config = rx_config.borrow().clone();

        // B. FPS Calculation & Write it to the frame
        current_fps = compute_actual_fps(&mut last_fps_time, current_fps);

        let fps_label = format!("FPS: {:.2}", current_fps);
        let _ = opencv::imgproc::put_text(
            &mut frame,
            &fps_label,
            Point::new(20, 40),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar::new(0.0, 255.0, 255.0, 0.0), // YELLOW IN BGR format
            2,
            opencv::imgproc::LINE_8,
            false,
        );

        // C. Draw the configurable virtual gate line across the frame
        let line_start = Point::new(0, current_config.gate_line_y);
        let line_end = Point::new(frame.cols(), current_config.gate_line_y);
        let _ = opencv::imgproc::line(
            &mut frame,
            line_start,
            line_end,
            Scalar::new(0.0, 0.0, 255.0, 0.0), // Red line for the gate boundary
            2,
            opencv::imgproc::LINE_8,
            0,
        );

        // D. Run YOLO inference
        let should_run_inference = frame_count % 1 == 0;
        if should_run_inference {
            let objects = vision_system.detect_vehicles(&frame);
            write_num_objects_detected_to_frame(&mut frame, objects.len());

            for obj in &objects {
                if is_a_vehicle(obj) {
                    process_vehicle_object(&mut frame, obj, &current_config);
                }
            }
        }

        // E. Send the frame to Web Server Broadcast Hub
        let mut encoded_buf = Vector::<u8>::new();
        let mut params = Vector::<i32>::new(); // Default compression params
        params.push(imgcodecs::IMWRITE_JPEG_QUALITY);
        params.push(75);
        if imgcodecs::imencode(".jpg", &frame, &mut encoded_buf, &params).unwrap_or(false) {
            let _ = stream_tx_cam1.send(encoded_buf.to_vec());
        }

        // F. Non-blocking cooldown logic for ALPR
        if is_alpr_active {
            if last_alpr_trigger.elapsed() >= Duration::from_secs(3) {
                is_alpr_active = false;
                println!("Camera 1: ALPR Cooldown finished.");
            }
        }
    }
}

fn compute_actual_fps(last_fps_time: &mut Instant, current_fps: f64) -> f64 {
    let now = Instant::now();
    let elapsed = now.duration_since(*last_fps_time).as_secs_f64();
    *last_fps_time = now;

    // If elapsed > 0, return the smoothed FPS calculation.
    // Otherwise, just return the existing current_fps.
    if elapsed > 0.0 {
        (current_fps * 0.9) + ((1.0 / elapsed) * 0.1)
    } else {
        current_fps
    }
}

fn is_a_vehicle(object: &TrackedObject) -> bool {
    matches!(
        object.class_name.as_str(),
        "car" | "truck" | "bus" | "motorcycle" | "bicycle"
    )
}

fn is_a_person(object: &TrackedObject) -> bool {
    object.class_name == "person"
}

fn draw_object_bounding_box(frame: &mut Mat, object: &TrackedObject, current_config: &AppConfig) {
    let box_area = (object.width * object.height) as f64;
    let rect = Rect::new(object.x, object.y, object.width, object.height);

    let box_color = if box_area >= current_config.min_box_area {
        Scalar::new(0.0, 255.0, 0.0, 0.0) // Green Box for trackable vehicles
    } else {
        Scalar::new(255.0, 255.0, 0.0, 0.0) // Yellow box for others
    };

    let _ = opencv::imgproc::rectangle(frame, rect, box_color, 2, 1, 0);

    let label = format!("{}- {}x{} (Area:{})", object.class_name, object.width, object.height, box_area);
    let _ = opencv::imgproc::put_text(
        frame,
        &label,
        Point::new(object.x, (object.y - 10).max(10)),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        box_color,
        1,
        opencv::imgproc::LINE_8,
        false,
    );
}

fn process_vehicle_object(frame: &mut Mat, object: &TrackedObject, current_config: &AppConfig) {
    draw_object_bounding_box(frame, object, current_config);
}

fn write_num_objects_detected_to_frame(frame: &mut Mat, num_objects: usize){
    let count_label = format!("Total Objects: {}", num_objects);
    let _ = opencv::imgproc::put_text(
        frame,
        &count_label,
        Point::new(20, 80), // Placed just below the FPS counter (which is at Y:40)
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        1.0, // Font scale
        Scalar::new(255.0, 0.0, 255.0, 0.0), // Magenta text color
        2,   // Thickness
        opencv::imgproc::LINE_8,
        false,
    );
}