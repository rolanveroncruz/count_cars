#![allow(unused,unused_imports)]
pub mod hailo8;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
#[allow(suspicious_runtime_symbol_definitions)]
pub mod bindings;
mod tiny_yolov4;
mod yolov8_lp;

use std::sync::{Arc, Mutex};
use opencv::core::Mat;
use crate::config::AppConfig;

// The unified struct used across the tracking pipeline
pub struct TrackedObject {
    pub id: usize,
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}
pub struct DetectedLicensePlate{
    pub confidence: f32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

pub trait CountCarsIntelligence: Send{
    fn detect_vehicles(&mut self, frame: &Mat) -> Vec<TrackedObject>;
    fn locate_plate(&mut self, cropped_vehicle: &Mat) -> Option<DetectedLicensePlate>;
    fn recognize_text(&mut self, cropped_plate: &Mat) -> Option<String>;
}

// Implement the trait for the thread-safe wrapper so it can be passed around cleanly
impl<T: CountCarsIntelligence> CountCarsIntelligence for Arc<Mutex<T>> {
    fn detect_vehicles(&mut self, frame: &Mat) -> Vec<TrackedObject> {
        self.lock().unwrap().detect_vehicles(frame)
    }

    fn locate_plate(&mut self, cropped_vehicle: &Mat) -> Option<DetectedLicensePlate> {
        self.lock().unwrap().locate_plate(cropped_vehicle)
    }

    fn recognize_text(&mut self, cropped_plate: &Mat) -> Option<String> {
        self.lock().unwrap().recognize_text(cropped_plate)
    }
}