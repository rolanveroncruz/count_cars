use opencv::core::Mat;
use crate::config::AppConfig;
use super::{ObjectDetector, TrackedObject};
pub struct Hailo8Inference{

}

impl Hailo8Inference {
    pub fn new() -> Self {
        Self {}
    }
}

impl ObjectDetector for Hailo8Inference {
    fn run_inference(&mut self, frame: &Mat, config: &AppConfig) -> Vec<TrackedObject> {
        todo!()
    }
}