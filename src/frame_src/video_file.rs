use opencv::core::Mat;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};
use std::time::{Instant, Duration};
use std::thread;
use super::FrameSource;
use opencv::prelude::*;

pub struct VideoFileSource {
    capture: videoio::VideoCapture,
    fps: f64,
    width: i32,
    height: i32,
    last_frame_time: Option<Instant>,
    frame_duration: Duration,
}

impl VideoFileSource {
    pub fn new(file_path: &str) -> Result<Self, String> {
        // Initialize OpenCV VideoCapture for the file
        let capture = videoio::VideoCapture::from_file(file_path, videoio::CAP_ANY)
            .map_err(|e| format!("Failed to open video file {}: {:?}", file_path, e))?;

        // Extract native metadata from the video file
        let raw_fps = capture.get(videoio::CAP_PROP_FPS).unwrap_or(30.0);
        let fps = if raw_fps <= 0.0 { 30.0 } else { raw_fps }; // Fallback in case of missing metadata

        let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(640.0) as i32;
        let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(640.0) as i32;

        // Calculate the required budget per frame to simulate real-time playback
        let frame_duration = Duration::from_secs_f64(1.0 / fps);

        Ok(Self {
            capture,
            fps,
            width,
            height,
            last_frame_time: None,
            frame_duration,
        })
    }
}

impl FrameSource for VideoFileSource {
    fn next_frame(&mut self) -> Option<Mat> {
        let mut frame = Mat::default();

        // Attempt to read the next frame
        if match self.capture.read(&mut frame) { Ok(true) => true, _ => false } {
            if frame.empty() {
                return None; // End of file
            }

            // --- SELF PACING LOGIC ---
            let now = Instant::now();
            if let Some(last) = self.last_frame_time {
                let elapsed = now.duration_since(last);
                if elapsed < self.frame_duration {
                    thread::sleep(self.frame_duration - elapsed);
                }
            }
            // Record the timestamp *after* the sleep so the next frame's budget is accurate
            self.last_frame_time = Some(Instant::now());

            Some(frame)
        } else {
            None // Failed to read or reached EOF
        }
    }

    fn fps(&self) -> f64 {
        self.fps
    }

    fn frame_size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    fn restart(&mut self) -> bool {
        // Reset the video pointer to frame 0
        self.capture.set(videoio::CAP_PROP_POS_FRAMES, 0.0).unwrap_or(false)
    }
}