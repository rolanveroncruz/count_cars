pub mod inference;
mod config;
mod alpr;
mod archiver;
mod camera;
mod server;

use alpr::GateEvent;
use tokio::sync::{watch, broadcast};
use std::sync::Arc;

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

    // 1e. Spawn Camera 2: The On-demand ALPR worker
    tokio::spawn(async move {
        alpr::run_alpr_worker(gate_rx_cam2).await;
    });

    // ✅ 1f. Spawn Background File Archiver
    tokio::spawn(async move {
        archiver::run_archiver(stream_rx_recorder).await;
    });

    // ✅ 1g. Spawn Camera 1: High Speed Yolo/ByteTrack Loop
    let gate_tx_cam1 = gate_tx.clone();
    let stream_tx_cam1 = stream_tx.clone();
    std::thread::spawn(move || {
        let yolo_engine = inference::hailo8::Hailo8Inference:: new();
        camera::run_camera_loop(config_rx, gate_tx_cam1, stream_tx_cam1, yolo_engine);
    });

    // ✅ 1h. AXUM WEB SERVER
    let state = server::AppState {
        config_tx: Arc::new(config_tx),
        stream_tx,
    };

    server::run_server(state).await?; // ✅ Awaits the extracted server function

    Ok(())
}