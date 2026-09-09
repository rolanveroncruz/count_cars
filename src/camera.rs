#![allow(unused)]
use tokio::sync::{watch, broadcast};
use std::process::{Command, Stdio};
use std::io::Read;
use crate::config::AppConfig;
use crate::alpr::GateEvent;
use trackforge::trackers::byte_track::ByteTrack;
use std::time::{Instant, Duration};

use opencv::{
    core::{Mat, Point, Scalar, Vector, Rect},
    imgcodecs,
};
use opencv::core::MatTraitConst;
use opencv::prelude::*;
use crate::inference::{ObjectDetector, TrackedObject};


pub fn run_camera_loop<D: ObjectDetector>(
    mut rx_config: watch::Receiver<AppConfig>,
    gate_tx_cam1: broadcast::Sender<GateEvent>,
    stream_tx_cam1: broadcast::Sender<Vec<u8>>,
    mut detector:D,
) {
    let mut is_alpr_active = false;
    let mut last_alpr_trigger = Instant::now() - Duration::from_secs(3);

    // Initialize the real ByteTrack engine.
    // tracker is really only used in: tracker.update(trackforge_detections).
    let mut tracker = ByteTrack::new(0.5, 30, 0.8, 0.6);

    println!("Starting High-Speed Camera/YOLO Loop...");

    // Initialize the physical camera (Camera 0-Sentinel)
    // This is basically like running 'rpicam-vide --camera 0 --t 0 --width ...' on the command line
    // and capturing the standard output.
    let mut child = match Command::new("rpicam-vid")
        .args(&[
            "--camera", "0",
            "-t", "0",
            "--width", "640",
            "--height", "640",
            "--vflip",
            "--framerate", "30",
            "--codec", "mjpeg", // compressed jpeg format.
            "-o", "-"
        ])
        .stdout(Stdio::piped())
        .spawn(){
        Ok(c)=> c,
        Err(e) => {
            eprintln!("Failed to spawn Camera 0 process: {:?}", e);
            return;
        }
    };
    let stdout = child.stdout.take() .take().unwrap();

    // prepare buffers to buffer the output and save these per frame.
    let mut reader = std::io::BufReader::new(stdout);
    let mut buffer: Vec<u8> = Vec::new(); // since jpeg is of variable length, we'll use a Vec to hold it.
    let mut small_buf = [0u8; 4096];    // 4kb is the typical size of a memory page in linux,
                                                  // which makes transferring efficient.

    let mut frame_count = 0;

    loop {
        // reader.read() reads the stdout into small_buf, and return the number of bytes read,
        // or an error.
        match reader.read(&mut small_buf) {
            Ok(0) => break, // EOF, camera process died
            Ok(n)=> {
                // If we read a valid amount of data, append it to the buffer.
                buffer.extend_from_slice(&small_buf[..n]);

                // Search for JPEG Start (0xff, 0xD8) and End (0xFF, 0xD9) markers
                // the let buffer.windows(w).position(|w|w==[xx,yy]) returns a Some(x) when the closure in position() returns true.
                // If it consumes all the windows, and none pass .position(), it returns None.
                while let Some(start) = buffer.windows(2).position(|w| w == [0xFF, 0xD8]) {
                    if let Some(end) = buffer[start..].windows(2).position(|w| w == [0xFF, 0xD9]) {
                        let end_idx = start + end + 2;

                        // ==============================================
                        //
                        // WE NOW HAVE A FULL PHYSICAL FRAME. PROCESS IT
                        //
                        //===============================================
                        let jpeg_data = buffer[start..end_idx].to_vec();
                        frame_count += 1;

                        // A. Check for live configuration updates with zero lag
                        if rx_config.has_changed().unwrap_or(false) {
                            let new_config = rx_config.borrow_and_update().clone();
                            println!("Camera 1: Camera Thread received new config: {:?}", new_config);
                        }
                        let current_config = rx_config.borrow().clone();

                        // Only run inference every 10th frame (~3 FPS if input is 30 FPS)
                        // to significantly improve performance on CPU.
                        let should_run_inference = frame_count % 10 == 0;

                        //====================================================
                        // OPENCV/YOLO AI PIPELINE
                        //===================================================

                        // A. Decode the raw JPEG bytes into an OpenCV Mat (Raw Pixels)
                        let mut frame = Mat::default();
                        let buf_vector = Vector::<u8>::from_slice(&jpeg_data);
                        if let Ok(decoded) = imgcodecs::imdecode(&buf_vector, imgcodecs::IMREAD_COLOR) {
                            frame = decoded;
                        }

                        if frame.empty() {
                             buffer.drain(0..end_idx);
                             continue;
                        }

                        // B. Draw the configurable virtual gate line across the frame for visual feedback on the stream
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


                        // C. Run YOLO inference, get all objects and put bounding boxes over them
                        let recognized_objects = if should_run_inference {
                             detector.run_inference(&frame, &current_config)
                        } else {
                             Vec::new()
                        };

                        // heartbeat
                        if should_run_inference {
                            use std::io::Write;
                            print!(".");
                            let _= std::io::stdout().flush();
                        }
                        
                        // D. Separate YOLO objects: Tracker Detections vs. Background Objects
                        let mut trackforge_detections = Vec::new();

                        if should_run_inference {
                            for object in &recognized_objects {
                                let box_area = (object.width * object.height) as f64;

                                if box_area <current_config.min_box_area{
                                    continue;
                                }
                                // If object is large enough and is a vehicle, add it to trackforge_detections for further processing;
                                // otherwise, just draw a bounding box around it.

                                if is_a_vehicle(&object) {
                                    trackforge_detections.push((
                                        [object.x as f32, object.y as f32, object.width as f32, object.height as f32],
                                        object.confidence,
                                        object.class_id as i64
                                    ));
                                } else if is_a_person(&object){
                                        draw_object_bounding_box(&mut frame, &object, &current_config);
                                }
                            } // end of iterating over recognized_objects
                        }

                        let active_tracks = if should_run_inference {
                             tracker.update(trackforge_detections)
                        } else {
                             // On frames without inference, we still want to maintain the ByteTrack state.
                             // Passing an empty vector to update() effectively ages existing tracks.
                             tracker.update(Vec::new())
                        };

                        for tracked_object in active_tracks {
                            let vehicle = TrackedObject{
                                id: tracked_object.track_id as usize,
                                class_id: tracked_object.class_id as usize,
                                class_name: match tracked_object.class_id {
                                    2 => "car".to_string(),
                                    3 => "motorcycle".to_string(),
                                    5 => "bus".to_string(),
                                    7 => "truck".to_string(),
                                    _ => format!("vehicle_{}", tracked_object.class_id),
                                },
                                confidence: tracked_object.score,
                                x: tracked_object.tlwh[0] as i32,
                                y: tracked_object.tlwh[1] as i32,
                                width: tracked_object.tlwh[2] as i32,
                                height: tracked_object.tlwh[3] as i32,
                            };
                            process_vehicle_object(&mut frame, &vehicle, &current_config)
                        }

                        // D. Send the frame to Web Server Broadcast Hub
                        let mut encoded_buf = Vector::<u8>::new();
                        let mut params = Vector::<i32>::new(); // Default compression params
                        params.push(opencv::imgcodecs::IMWRITE_JPEG_QUALITY);
                        params.push(75);
                        if imgcodecs::imencode(".jpg", &frame, &mut encoded_buf, &params).unwrap_or(false) {
                            let _ = stream_tx_cam1.send(encoded_buf.to_vec());
                        }

                        // Non-blocking cooldown logic for ALPR
                        if is_alpr_active {
                            if last_alpr_trigger.elapsed() >= Duration::from_secs(3) {
                                is_alpr_active = false;
                                println!("Camera 1: ALPR Cooldown finished.");
                            }
                        }
                        buffer.drain(0..end_idx);
                    } else{
                        // We found a start marker but no end marker,
                        // break out of the while loop to go read more bytes from the camera
                        break;
                    }
                } // end of while to search for the start of the frame
            }// enf of Ok(n)
            _ => {}
        }// end of reading buffer
    }// enf of loop{}
} // end of fn run_camera_loop

fn is_a_vehicle(object: &TrackedObject)->bool{
    if object.class_name == "car" ||
        object.class_name == "truck" ||
        object.class_name == "bus" ||
        object.class_name == "motorcycle" ||
        object.class_name == "bicycle" {
        return true;
    }
    false
}

fn is_a_person(object: &TrackedObject)->bool{
    if object.class_name == "person" {
        return true;
    }
    false
}

fn draw_object_bounding_box(frame: &mut Mat, object: &TrackedObject, current_config: &AppConfig){
    let box_area = (object.width * object.height) as f64;
    let rect = Rect::new(object.x, object.y, object.width, object.height);

    // 1a. Draw a rectangle for EVERY object detected so the client sees everything.
    let box_color = if box_area >= current_config.min_box_area {
        Scalar::new(0.0, 255.0, 0.0, 0.0)  //Green Box for trackable vehicles
    } else {
        Scalar::new(255.0, 255.0, 0.0, 0.0) // Yellow box for others
    };

    let _result = opencv::imgproc::rectangle(
        frame,
        rect,
        box_color,
        2,  // thickness
        1,  // line_type
        0,     // shift
    );


    // 1b.Render the class name and confidence label above the bounding box
    let label = format!("{}: {:.2}", object.class_name, object.confidence);
    let _result = opencv::imgproc::put_text(
        frame,
        &label,
        Point::new(object.x, (object.y - 10).max(10)),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        box_color,
        1,
        opencv::imgproc::LINE_8,
        false
    );

} // end of draw_object_bounding_box

// process_vehicle_object takes the frame, the detected vehicle, and the current configuration.
// We can assume it is large enough to be trackable.
fn process_vehicle_object(frame: &mut Mat, object: &TrackedObject, current_config: &AppConfig){
    draw_object_bounding_box(frame, object, &current_config);
}