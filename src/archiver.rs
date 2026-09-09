// ✅✅✅ MULTIPLE LINE CHANGES BELOW ✅✅✅
// ✅ New file for background file archiver task
use tokio::sync::broadcast::{self, error::RecvError};
use std::fs::{self, File};
use std::io::Write;
use chrono::Local;

pub async fn run_archiver(mut rx_rec: broadcast::Receiver<Vec<u8>>) {

    if let Err(e) = fs::create_dir_all("archives") {
        eprintln!("[Recorder] Failed to create archive directory: {:?}", e);
        return;
    }
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("archives/archive_{}.mjpeg", timestamp);
    println!("[Recorder] Background file archiver started. Saving to {}.", filename);

    let mut archive_file = match File::create(filename) {
        Ok(file) => Some(file),
        Err(e) => {
            eprintln!("[Recorder] Failed to create archive file: {:?}", e);
            None
        }
    };
    loop {
        match rx_rec.recv().await{
            Ok(frame_data)=> {
                if let Some (ref mut file)=archive_file{
                    let _ = file.write_all(&frame_data);
                }
            }
            Err(RecvError::Lagged(_)) => {
                eprintln!("[Recorder] Lagged behind in receiving frames. Continuing...");
            }
            Err(RecvError::Closed) => {
                println!("[Recorder] Broadcast channel closed, stopping archiver");
                break;
            }
        }
    }
}
