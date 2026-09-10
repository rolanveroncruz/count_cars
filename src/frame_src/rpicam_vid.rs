#![allow(unused,unused_imports)]
use std::process::{Command, Stdio, Child, ChildStdout};
use std::io::{BufReader, Read};
use opencv::core::{Mat, Vector};
use opencv::prelude::*;
use opencv::imgcodecs;
use super::FrameSource;

pub struct RpiCamVid {
    child: Child,
    reader: BufReader<ChildStdout>,
    buffer: Vec<u8>,
    small_buf: [u8; 4096],
    width: i32,
    height: i32,
    fps: f64,
}

impl RpiCamVid {
    pub fn new(width: i32, height: i32, fps: f64) -> Result<Self, String> {
        // Initialize the physical camera as a background subprocess
        let mut child = Command::new("rpicam-vid")
            .args(&[
                "--camera", "0",
                "-t", "0",
                "--width", &width.to_string(),
                "--height", &height.to_string(),
                "--vflip",
                "--framerate", &fps.to_string(),
                "--codec", "mjpeg", // Hardware compressed jpeg format
                "-o", "-"           // Stream to standard output
            ])
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn rpicam-vid process: {:?}", e))?;

        let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
        let reader = BufReader::new(stdout);

        Ok(Self {
            child,
            reader,
            buffer: Vec::new(),
            small_buf: [0u8; 4096], // 4KB Linux memory page size for efficiency
            width,
            height,
            fps,
        })
    }
}

impl FrameSource for RpiCamVid {
    fn next_frame(&mut self) -> Option<Mat> {
        loop {
            // Read from the stdout pipe into the small buffer
            match self.reader.read(&mut self.small_buf) {
                Ok(0) => return None, // EOF, camera process died
                Ok(n) => {
                    self.buffer.extend_from_slice(&self.small_buf[..n]);

                    // Search for JPEG Start (0xFF, 0xD8) and End (0xFF, 0xD9) markers
                    while let Some(start) = self.buffer.windows(2).position(|w| w == [0xFF, 0xD8]) {
                        if let Some(end) = self.buffer[start..].windows(2).position(|w| w == [0xFF, 0xD9]) {
                            let end_idx = start + end + 2;
                            let jpeg_data = self.buffer[start..end_idx].to_vec();

                            // Remove the processed bytes from the buffer
                            self.buffer.drain(0..end_idx);

                            // Decode the raw JPEG bytes into an OpenCV Mat
                            let buf_vector = Vector::<u8>::from_slice(&jpeg_data);
                            if let Ok(decoded) = imgcodecs::imdecode(&buf_vector, imgcodecs::IMREAD_COLOR) {
                                if !decoded.empty() {
                                    return Some(decoded);
                                }
                            }
                        } else {
                            // CRITICAL FIX: We have a start marker but no end marker yet.
                            // Break out of the while loop to read more bytes from the camera!
                            break;
                        }
                    }
                }
                Err(_) => return None, // Pipe broken
            }
        }
    }

    fn fps(&self) -> f64 {
        self.fps
    }

    fn frame_size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    fn restart(&mut self) -> bool {
        // A live physical camera stream generally cannot be instantly "looped" like a file.
        // If it dies, the system should ideally handle re-instantiating the struct.
        false
    }
}