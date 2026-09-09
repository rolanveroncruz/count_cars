pub mod hailo8;

use opencv::core::Mat;
use crate::config::AppConfig;
pub struct TrackedObject{
    pub id: usize,
    pub class_id:usize,
    pub class_name:String,
    pub confidence:f32,
    pub x:i32,
    pub y:i32,
    pub width:i32,
    pub height:i32,
}

pub trait ObjectDetector: Send{
    fn run_inference(&mut self, frame: &Mat, config:&AppConfig)->Vec<TrackedObject>;
}
