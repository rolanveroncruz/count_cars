pub mod inference;
mod config;
mod alpr;
mod archiver;
mod camera;
mod server;
mod frame_src;

use alpr::GateEvent;
use tokio::sync::{watch, broadcast};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1a. configuration
    let initial_config = config::get_or_create_initial_config();
    println!("Starting system with config: {:?}", initial_config);

    // 1b. Set up the watch channel for real-time dashboard config updates
    let (config_tx, config_rx) = watch::channel(initial_config);

    // 1c. Set up the Broadcast channel for camera triggers (capacity of 16)
    let (gate_tx, gate_rx_cam2) = broadcast::channel::<GateEvent>(16);

    // 1d. Set up Central Video Broadcast Hub (capacity of 30 frames)
    let (stream_tx, stream_rx_recorder) = broadcast::channel::<Vec<u8>>(30);

    // 1e. Setup Hailo8Manager
    let hailo8_manager = Arc::new(Mutex::new(inference::hailo8::Hailo8Manager::new(
        "src/inference/models/yolov8n.hef",
          "src/inference/models/tiny_yolov4_license_plates.hef",
           "src/inference/models/lprnet.hef",
    )));
    // 1e. Spawn Camera 2: The On-demand ALPR worker
    tokio::spawn(async move {
        alpr::run_alpr_worker(gate_rx_cam2).await;
    });

    // ✅ 1f. Spawn Background File Archiver
    tokio::spawn(async move {
        archiver::run_archiver(stream_rx_recorder).await;
    });

    // 1f. Read command line params to determine frame source
    let mut args = std::env::args().skip(1);
    let mut source_arg = String::from(" camera"); // Defaults to camera
    while let Some(arg) = args.next(){
        if arg == "--source" {
            if let Some(val) = args.next(){
                source_arg = val;
            }
        }
    }
    println!("Frame source set to: {}", source_arg);

    // ✅ 1g. Spawn Camera 1: High Speed Yolo/ByteTrack Loop
    let gate_tx_cam1 = gate_tx.clone();
    let stream_tx_cam1 = stream_tx.clone();
    let hailo_mgr = Arc::clone(&hailo8_manager);

    std::thread::spawn(move || {
        if source_arg.starts_with("file:"){
            //Option A : Read from a saved video file
            let file_path = &source_arg.trim_start_matches("file:");
            let video_file = frame_src::video_file::VideoFileSource::new(file_path).expect("Failed to create VideoFile instance");
            camera::run_camera_loop(config_rx, gate_tx_cam1, stream_tx_cam1, hailo_mgr, video_file);
        } else {
            let physical_camera = frame_src::rpicam_vid::RpiCamVid::new(
                640,
                640,
                30.0,
            ).expect("Failed to create RpiCamVid instance");
            camera::run_camera_loop(config_rx, gate_tx_cam1, stream_tx_cam1, hailo_mgr, physical_camera);
        }
    });

    // ✅ 1h. AXUM WEB SERVER
    let state = server::AppState {
        config_tx: Arc::new(config_tx),
        stream_tx,
    };

    server::run_server(state).await?; // ✅ Awaits the extracted server function

    Ok(())
}