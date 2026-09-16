 #![allow(dead_code)]
pub mod rpicam_vid;

pub mod video_file;

use opencv::core::Mat;

pub trait FrameSource {

    // Returns the next frame as a raw OpenCV Mat
    fn next_frame(&mut self) -> Option<Mat>;

    // Compute the fps.
    fn fps(&self) -> f64;

    // Return the frame size
    fn frame_size(&self) -> (i32, i32);

    // a restart function will allow us to loop a short video infinitely
    fn restart(&mut self)->bool;
}
