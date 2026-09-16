#![allow(unused,unused_imports)]
use opencv::core::Mat;
use opencv::prelude::*;
use crate::config::AppConfig;
use super::{  CountCarsIntelligence, TrackedObject};
use std::ffi::CString;
use std::ptr;
use crate::inference::bindings::*;

const COCO_NAMES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

pub struct Hailo8Manager {
    vdevice: hailo_vdevice,
    vehicle_hef: hailo_hef,
    plate_hef: hailo_hef,
    ocr_hef: hailo_hef,
    // Vehicle YOLO pipeline (configured at init, reused per-frame)
    vehicle_network_group: hailo_configured_network_group,
    vehicle_input_vstream: hailo_input_vstream,
    vehicle_input_size: usize,
    vehicle_output_vstreams: Vec<hailo_output_vstream>,
    vehicle_output_sizes: Vec<usize>,
    vehicle_is_nms: bool,
}

// We must explicitly tell Rust it is safe to move these C-pointers across thread boundaries
unsafe impl Send for Hailo8Manager {}

impl Hailo8Manager {
    pub fn new(vehicle_path: &str, plate_path: &str, ocr_path: &str) -> Self {
        println!("Initializing Central Hailo-8 Hardware Manager...");

        // ✅ 1. Check if the Vehicle HEF file exists
        if !std::path::Path::new(vehicle_path).exists() {
            panic!("Vehicle HEF file not found at: {}", vehicle_path);
        }
        println!("Vehicle HEF file found at: {}", vehicle_path);

        // ✅ 2. Check if the Plate HEF file exists
        if !std::path::Path::new(plate_path).exists() {
            panic!("Plate HEF file not found at: {}", plate_path);
        }
        println!("Plate HEF file found at: {}", plate_path);

        // ✅ 3. Check if the OCR HEF file exists
        if !std::path::Path::new(ocr_path).exists() {
            panic!("OCR HEF file not found at: {}", ocr_path);
        }
        println!("OCR HEF file found at: {}", ocr_path);

        let mut vdevice: hailo_vdevice = ptr::null_mut();
        let mut vehicle_hef: hailo_hef = ptr::null_mut();
        let mut plate_hef: hailo_hef = ptr::null_mut();
        let mut ocr_hef: hailo_hef = ptr::null_mut();


        let mut network_group: hailo_configured_network_group = ptr::null_mut();
        let mut input_vstream: hailo_input_vstream = ptr::null_mut();
        let mut input_size: usize = 0;
        let mut output_vstream_ptrs: Vec<hailo_output_vstream> = Vec::new();
        let mut output_sizes: Vec<usize> = Vec::new();
        let mut is_nms = false;

        unsafe {
            println!("Step 1: Initialize the Virtual Device (VDevice) Params");
            // 1. Initialize the Virtual Device (VDevice) Params
            let mut vdevice_params: hailo_vdevice_params_t = std::mem::zeroed();
            let status = hailo_init_vdevice_params(&mut vdevice_params);
            if status != HAILO_SUCCESS {
                panic!("Failed to init VDevice params: {}", status);
            }

            println!("Step 2: Create the VDevice connection over PCIe");
            // 2. Create the VDevice connection over PCIe
            let status = hailo_create_vdevice(&mut vdevice_params, &mut vdevice);
            if status != HAILO_SUCCESS {
                panic!("Failed to create Hailo VDevice: {}", status);
            }

            println!("Step 3: Load the three HEF files into the hardware memory");
            // 3. Load the three HEF files into the hardware memory
            let v_path = CString::new(vehicle_path).unwrap();
            if hailo_create_hef_file(&mut vehicle_hef, v_path.as_ptr()) != HAILO_SUCCESS {
                panic!("Failed to load vehicle HEF");
            }

            let p_path = CString::new(plate_path).unwrap();
            if hailo_create_hef_file(&mut plate_hef, p_path.as_ptr()) != HAILO_SUCCESS {
                panic!("Failed to load plate HEF");
            }

            let o_path = CString::new(ocr_path).unwrap();
            if hailo_create_hef_file(&mut ocr_hef, o_path.as_ptr(), ) != HAILO_SUCCESS {
                panic!("Failed to load OCR HEF");
            }
            println!("Step 4: Configure the vehicle HEF on the VDevice");
            //4. Configure the vehicle HEF on the VDevice
            let mut configure_params: hailo_configure_params_t = std::mem::zeroed();
            if hailo_init_configure_params_by_vdevice(vehicle_hef, vdevice, &mut configure_params) != HAILO_SUCCESS {
                panic!("Failed to init configure params for vehicle HEF");
            }

            println!("Hailo-8 Manager successfully loaded all 3 models!");

            let mut num_network_groups: usize = 1;
            if hailo_configure_vdevice(
                vdevice,
                vehicle_hef,
                &mut configure_params,
                &mut network_group,
                &mut num_network_groups
            ) != HAILO_SUCCESS {
                panic!("Failed to configure vehicle HEF on VDevice");
            }
            println!("Vehicle network group configured (count={})", num_network_groups);

            // 5. Query all vstream infos from the HEF (HailoRT 4.23.0 API)
            let mut all_vstream_infos: [hailo_vstream_info_t; HAILO_MAX_STREAMS_COUNT as usize] = std::mem::zeroed();
            let mut num_all_vstreams: usize = HAILO_MAX_STREAMS_COUNT as usize;
            if hailo_hef_get_all_vstream_infos(
                vehicle_hef,
                ptr::null(), // NULL = default network group name
                all_vstream_infos.as_mut_ptr(),
                &mut num_all_vstreams,
            ) != HAILO_SUCCESS {
                panic!("Failed to get vstream infos from vehicle HEF");
            }

            // Separate into input (H2D) and output (D2H) infos
            let mut h2d_infos: Vec<hailo_vstream_info_t> = Vec::new();
            let mut d2h_infos: Vec<hailo_vstream_info_t> = Vec::new();
            for i in 0..num_all_vstreams {
                let info = ptr::read(&all_vstream_infos[i]);
                if info.direction == HAILO_H2D_STREAM {
                    h2d_infos.push(info);
                } else {
                    d2h_infos.push(info);
                }
            }
            println!("Vehicle HEF: {} input(s), {} output(s)", h2d_infos.len(), d2h_infos.len());

            // 6. Build input vstream params (HailoRT 4.23.0 API)
            let mut input_params: [hailo_input_vstream_params_by_name_t; HAILO_MAX_STREAMS_COUNT as usize] = std::mem::zeroed();
            let mut input_params_count: usize = HAILO_MAX_STREAMS_COUNT as usize;
            if hailo_make_input_vstream_params(
                network_group,
                true,
                HAILO_FORMAT_TYPE_AUTO,
                input_params.as_mut_ptr(),
                &mut input_params_count,
            ) != HAILO_SUCCESS {
                panic!("Failed to make input vstream params");
            }
            // 7. Build output vstream params (HailoRT 4.23.0 API)
            let mut output_params: [hailo_output_vstream_params_by_name_t; HAILO_MAX_STREAMS_COUNT as usize] = std::mem::zeroed();
            let mut output_params_count: usize = HAILO_MAX_STREAMS_COUNT as usize;

            if hailo_make_output_vstream_params(
                network_group,
                false,
                HAILO_FORMAT_TYPE_FLOAT32,
                output_params.as_mut_ptr(),
                &mut output_params_count,
            ) != HAILO_SUCCESS {
                panic!("Failed to make output vstream params");
            }
            // 8. Create input vstreams
            if hailo_create_input_vstreams(
                network_group,
                input_params.as_ptr(),
                input_params_count,
                &mut input_vstream,
            ) != HAILO_SUCCESS {
                panic!("Failed to create vehicle input vstreams");
            }

            // 9. Create output vstreams
            output_vstream_ptrs = vec![ptr::null_mut(); d2h_infos.len()];
            if hailo_create_output_vstreams(
                network_group,
                output_params.as_ptr(),
                output_params_count,
                output_vstream_ptrs.as_mut_ptr(),
            ) != HAILO_SUCCESS {
                panic!("Failed to create vehicle output vstreams");
            }

            // 10. Compute input size and output sizes for buffer allocation
            let mut expected_in_size:usize = 0;
            if hailo_get_input_vstream_frame_size(input_vstream, &mut expected_in_size) != HAILO_SUCCESS {
                panic!("Failed to get vehicle input vstream frame size");
            }
            input_size = expected_in_size;

            is_nms = false;

            for i in 0..d2h_infos.len(){
                let order = d2h_infos[i].format.order;

                match order{
                    HAILO_FORMAT_ORDER_HAILO_NMS => {println!("Vehicle Output {} Format:LEGACY_NMS", i);}
                    HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS => println!("Vehicle Output {} Format: NMS_BY_CLASS", i),
                    HAILO_FORMAT_ORDER_HAILO_NMS_BY_SCORE => println!("Vehicle Output {} Format: NMS_BY_SCORE", i),
                    _ => println!("Vehicle Output {} Format: RAW TENSOR (Enum ID: {})", i, order),
                }
                if order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS{
                    is_nms = true;
                }
                let mut expected_out_size:usize = 0;
                if hailo_get_output_vstream_frame_size(output_vstream_ptrs[i], &mut expected_out_size) != HAILO_SUCCESS {
                    panic!("Failed to get vehicle output vstream frame size");
                }
                output_sizes.push(expected_out_size);
            }

            println!("Vehicle input size: {}, input_size, NMS output: {}", input_size, is_nms);
        } // unsafe

        Self {
            vdevice,
            vehicle_hef,
            plate_hef,
            ocr_hef,
            vehicle_network_group: network_group,
            vehicle_input_vstream: input_vstream,
            vehicle_input_size: input_size,
            vehicle_output_vstreams: output_vstream_ptrs,
            vehicle_output_sizes: output_sizes,
            vehicle_is_nms: is_nms,
        }
    }
}

// We must manually clean up the C memory when this struct goes out of scope
impl Drop for Hailo8Manager {
    fn drop(&mut self) {
        unsafe {

            // Release vehicle pipeline vstreams
            // Release input vstreams (batch API)
            if !self.vehicle_input_vstream.is_null() {
                hailo_release_input_vstreams(
                    &mut self.vehicle_input_vstream,
                    1,
                );
            }
            // Release output vstreams (batch API)
            if !self.vehicle_output_vstreams.is_empty() {
                hailo_release_output_vstreams(
                    self.vehicle_output_vstreams.as_mut_ptr(),
                    self.vehicle_output_vstreams.len(),
                );
            }
            // Release HEFs and VDevice
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
        let mut objects = Vec::new();
        if frame.empty() { return objects; }

        // ==========================================
        // 1. PREPROCESS OPENCV MAT (640X640 RGB)
        // ==========================================
        let mut resized  = Mat::default();
        let size = opencv::core::Size::new(640,640);
        let _ = opencv::imgproc::resize(
            frame,
            &mut resized,
            size,
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        );
        let mut rgb_frame = Mat::default();
        let _ = opencv::imgproc::cvt_color_def(
            &resized,
            &mut rgb_frame,
            opencv::imgproc::COLOR_BGR2RGB,
        );
        println!("Inference frame: {}x{} channels={} bytes={}",
        rgb_frame.cols(),
        rgb_frame.rows(),
        rgb_frame.channels(),
        rgb_frame.data_bytes().unwrap().len()
        );
        // ==========================================
        // 2. EXTRACT BYTES AND RUN INFERENCE
        // ==========================================
        if let Ok(frame_data) = rgb_frame.data_bytes() {
            println!(
                "Input: {}x{}, channels={}, bytes={}",
                rgb_frame.cols(),
                rgb_frame.rows(),
                rgb_frame.channels(),
                frame_data.len()
            );

            let min = *frame_data.iter().min().unwrap();
            let max = *frame_data.iter().max().unwrap();
            let sum: u64 = frame_data.iter().map(|&x| x as u64).sum();
            let mean = sum as f64 / frame_data.len() as f64;

            println!(
                "Input statistics: min={}, max={}, mean={:.2}",
                min, max, mean
            );
            unsafe{
                let status = hailo_vstream_write_raw_buffer(
                    self.vehicle_input_vstream,
                    frame_data.as_ptr() as *const std::ffi::c_void,
                    self.vehicle_input_size,
                );
                if status != HAILO_SUCCESS {
                    eprintln!("Hailo input  write failed: {}", status);
                    return objects;
                }

            }
        } else {
            return objects;
        }

        // ==========================================
        // 3. READ OUTPUT FROM HAILO OUTPUT VSTREAMS
        // ==========================================
        let orig_h = frame.rows() as f32;
        let orig_w = frame.cols() as f32;

        for (i, output_vs) in self.vehicle_output_vstreams.iter().enumerate() {
            let buf_size = self.vehicle_output_sizes[i];
            let mut output_buf:Vec<u8> = vec![0u8; buf_size];

            unsafe{
                let status = hailo_vstream_read_raw_buffer(
                    *output_vs,
                    output_buf.as_mut_ptr() as *mut std::ffi::c_void,
                    buf_size,
                );
                let float_count = output_buf.len() / size_of::<f32>();

                let output_floats = unsafe {
                    std::slice::from_raw_parts(
                        output_buf.as_ptr() as *const f32,
                        float_count,
                    )
                };

                let mut max_value = f32::NEG_INFINITY;
                let mut min_value = f32::INFINITY;
                let mut nonzero = 0usize;

                for &v in output_floats {
                    if v != 0.0 {
                        nonzero += 1;
                    }
                    max_value = max_value.max(v);
                    min_value = min_value.min(v);
                }

                println!(
                    "NMS raw output: floats={} nonzero={} min={} max={}",
                    float_count, nonzero, min_value, max_value
                );

                println!(
                    "NMS first 32 floats: {:?}",
                    &output_floats[..32.min(output_floats.len())]
                );
                println!("in detect_vehicles(): Vehicle output {}: {} bytes", i, output_buf.len());
                if status != HAILO_SUCCESS{
                    eprintln!("Hailo output read failed on vstream {}:{}", i, status);
                    continue;
                }
            }

            // ==========================================
            // 4. PARSE NMS DETECTIONS INTO TrackedObject
            // ==========================================
            if self.vehicle_is_nms {
                parse_nms_output(&output_buf, orig_w, orig_h, &mut objects);
            }
        }
        println!("in detect_vehicles(): parsed {} objects", objects.len());
        objects
    } // detect_vehicles()

    fn locate_plate(&mut self, cropped_vehicle: &Mat) -> Option<TrackedObject> {
        if cropped_vehicle.empty() { return None; }
        todo!()
    }

    fn recognize_text(&mut self, cropped_plate: &Mat) -> Option<String> {
        if cropped_plate.empty() { return None; }
        todo!()
    }
}
fn parse_nms_output(
    buffer: &[u8],
    orig_w: f32,
    orig_h: f32,
    objects: &mut Vec<TrackedObject>,
) {
    // HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS layout:
    //
    // For each class:
    //
    //   struct (packed) {
    //       float32_t bbox_count;
    //       hailo_bbox_float32_t bbox[bbox_count];
    //   };
    //
    // Each bbox contains:
    //
    //   y_min, x_min, y_max, x_max, score
    //
    // Each value is a float32 (20 bytes per bbox).
    //
    // IMPORTANT:
    // There are not max_bboxes_per_class entries stored for every
    // class. The buffer contains only bbox_count actual entries.

    let num_classes = 80; // COCO

    let mut obj_id = 0;
    let mut offset = 0usize;

    for class_id in 0..num_classes {
        // Need at least 4 bytes for bbox_count.
        if offset + 4 > buffer.len() {
            break;
        }

        // bbox_count is a float32.
        let bbox_count = f32::from_le_bytes([
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]) as usize;

        offset += 4;

        if class_id < 10 {
            println!(
                "NMS class {} ({}) count={}",
                class_id,
                COCO_NAMES.get(class_id).unwrap_or(&"unknown"),
                bbox_count
            );
        }

        let class_name = COCO_NAMES
            .get(class_id)
            .unwrap_or(&"unknown")
            .to_string();

        // Each bbox is:
        //
        // y_min: 4 bytes
        // x_min: 4 bytes
        // y_max: 4 bytes
        // x_max: 4 bytes
        // score: 4 bytes
        //
        // = 20 bytes
        for _ in 0..bbox_count {
            if offset + 20 > buffer.len() {
                println!(
                    "WARNING: NMS buffer ended while parsing class {} \
                     (wanted bbox at offset {}, buffer size {})",
                    class_id,
                    offset,
                    buffer.len()
                );
                return;
            }

            let y_min = f32::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
            ]);

            let x_min = f32::from_le_bytes([
                buffer[offset + 4],
                buffer[offset + 5],
                buffer[offset + 6],
                buffer[offset + 7],
            ]);

            let y_max = f32::from_le_bytes([
                buffer[offset + 8],
                buffer[offset + 9],
                buffer[offset + 10],
                buffer[offset + 11],
            ]);

            let x_max = f32::from_le_bytes([
                buffer[offset + 12],
                buffer[offset + 13],
                buffer[offset + 14],
                buffer[offset + 15],
            ]);

            let score = f32::from_le_bytes([
                buffer[offset + 16],
                buffer[offset + 17],
                buffer[offset + 18],
                buffer[offset + 19],
            ]);

            offset += 20;

            let x = (x_min * orig_w) as i32;
            let y = (y_min * orig_h) as i32;
            let w = ((x_max - x_min) * orig_w) as i32;
            let h = ((y_max - y_min) * orig_h) as i32;

            objects.push(TrackedObject {
                id: obj_id,
                class_id,
                class_name: class_name.clone(),
                confidence: score,
                x,
                y,
                width: w,
                height: h,
            });

            obj_id += 1;
        }
    }

    println!(
        "NMS parser: parsed {} objects, consumed {} / {} bytes",
        obj_id,
        offset,
        buffer.len()
    );
}