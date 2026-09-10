#![allow(unused,unused_imports)]
use opencv::core::Mat;
use opencv::prelude::*;
use crate::config::AppConfig;
use super::{  CountCarsIntelligence, TrackedObject};
use std::ffi::CString;
use std::ptr;
use hailort_sys::*;

pub struct Hailo8Manager {
    vdevice: hailo_vdevice,
    vehicle_hef: hailo_hef,
    plate_hef: hailo_hef,
    ocr_hef: hailo_hef,
}

// We must explicitly tell Rust it is safe to move these C-pointers across thread boundaries
unsafe impl Send for Hailo8Manager {}

impl Hailo8Manager {
    pub fn new(vehicle_path: &str, plate_path: &str, ocr_path: &str) -> Self {
        println!("Initializing Central Hailo-8 Hardware Manager...");

        // ✅ 1. Check if the Vehicle HEF file exists
        if !std::path::Path::new(vehicle_path).exists() {
            panic!("❌ Vehicle HEF file not found at: {}", vehicle_path);
        }
        println!("✅ Vehicle HEF file found at: {}", vehicle_path);

        // ✅ 2. Check if the Plate HEF file exists
        if !std::path::Path::new(plate_path).exists() {
            panic!("❌ Plate HEF file not found at: {}", plate_path);
        }
        println!("✅ Plate HEF file found at: {}", plate_path);

        // ✅ 3. Check if the OCR HEF file exists
        if !std::path::Path::new(ocr_path).exists() {
            panic!("❌ OCR HEF file not found at: {}", ocr_path);
        }
        println!("✅ OCR HEF file found at: {}", ocr_path);
        let mut vdevice: hailo_vdevice = ptr::null_mut();
        let mut vehicle_hef: hailo_hef = ptr::null_mut();
        let mut plate_hef: hailo_hef = ptr::null_mut();
        let mut ocr_hef: hailo_hef = ptr::null_mut();

        unsafe {
            // 1. Initialize the Virtual Device (VDevice) Params
            let mut vdevice_params: hailo_vdevice_params_t = std::mem::zeroed();
            let status = hailo_init_vdevice_params(&mut vdevice_params);
            if status != HAILO_SUCCESS {
                panic!("Failed to init VDevice params: {}", status);
            }

            // 2. Create the VDevice connection over PCIe
            let status = hailo_create_vdevice(&mut vdevice_params,  &mut vdevice);
            if status != HAILO_SUCCESS {
                panic!("Failed to create Hailo VDevice: {}", status);
            }

            // 3. Load the three HEF files into the hardware memory
            let v_path = CString::new(vehicle_path).unwrap();
            if hailo_create_hef_file(&mut vehicle_hef, v_path.as_ptr())  != HAILO_SUCCESS {
                panic!("Failed to load vehicle HEF");
            }

            let p_path = CString::new(plate_path).unwrap();
            if hailo_create_hef_file(&mut plate_hef, p_path.as_ptr()) != HAILO_SUCCESS {
                panic!("Failed to load plate HEF");
            }

            let o_path = CString::new(ocr_path).unwrap();
            if hailo_create_hef_file(&mut ocr_hef, o_path.as_ptr(),) != HAILO_SUCCESS {
                panic!("Failed to load OCR HEF");
            }
        }

        println!("✅ Hailo-8 Manager successfully loaded all 3 models!");

        Self {
            vdevice,
            vehicle_hef,
            plate_hef,
            ocr_hef,
        }
    }
}

// We must manually clean up the C memory when this struct goes out of scope
impl Drop for Hailo8Manager {
    fn drop(&mut self) {
        unsafe {
            if !self.ocr_hef.is_null() { hailo_release_hef(self.ocr_hef); }
            if !self.plate_hef.is_null() { hailo_release_hef(self.plate_hef); }
            if !self.vehicle_hef.is_null() { hailo_release_hef(self.vehicle_hef); }
            if !self.vdevice.is_null() { hailo_release_vdevice(self.vdevice); }
        }
    }
}

// =====================================================================
// TRAIT IMPLEMENTATIONS (The API used by your Camera/Worker threads)
// =====================================================================

impl CountCarsIntelligence for Hailo8Manager {
    fn detect_vehicles(&mut self, frame: &Mat) -> Vec<TrackedObject> {
        let objects = Vec::new();
        if frame.empty() { return objects; }

        // TODO:
        // 1. Convert `frame` to raw RGB flat array
        // 2. Write to `vehicle_hef` input vstream (hailo_vstream_write_raw_buffer)
        // 3. Read output vstream (hailo_vstream_read_raw_buffer)
        // 4. Parse bounding boxes

        objects
    }

    fn locate_plate(&mut self, cropped_vehicle: &Mat) -> Option<TrackedObject> {
        if cropped_vehicle.empty() { return None; }
        todo!()
    }

    fn recognize_text(&mut self, cropped_plate: &Mat) -> Option<String> {
        if cropped_plate.empty() { return None; }
        todo!()
    }
}
