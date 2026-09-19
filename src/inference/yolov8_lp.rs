use opencv::core::{Mat, MatTraitConst, MatTraitConstManual};
use crate::inference::bindings::{hailo_vstream_write_raw_buffer, HAILO_SUCCESS, hailo_vstream_read_raw_buffer};
use crate::inference::hailo8::Hailo8Manager;

impl Hailo8Manager{
    pub fn recognize_text(&mut self, cropped_plate: &Mat) -> Option<String> {
        if cropped_plate.empty() {
            return None;
        }

        // ---------------------------------------------------------
        // 1. Pre-process plate image for YOLOv8-LP
        // ---------------------------------------------------------

        let mut resized = Mat::default();

        let size = opencv::core::Size::new(256, 128);

        opencv::imgproc::resize(
            cropped_plate,
            &mut resized,
            size,
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )
            .ok()?;

        let mut rgb_frame = Mat::default();

        opencv::imgproc::cvt_color_def(
            &resized,
            &mut rgb_frame,
            opencv::imgproc::COLOR_BGR2RGB,
        )
            .ok()?;

        // ---------------------------------------------------------
        // 2. Send RGB image to Hailo
        // ---------------------------------------------------------

        let frame_data = rgb_frame.data_bytes().ok()?;

        if frame_data.len() != self.ocr_input_size {
            println!(
                "OCR: input size mismatch: got {}, expected {}",
                frame_data.len(),
                self.ocr_input_size
            );
            return None;
        }

        println!("OCR: About to write input");

        unsafe {
            let status = hailo_vstream_write_raw_buffer(
                self.ocr_input_vstream,
                frame_data.as_ptr() as *const std::ffi::c_void,
                self.ocr_input_size,
            );

            println!("OCR: Input written: {}", status);

            if status != HAILO_SUCCESS {
                return None;
            }
        }

        // ---------------------------------------------------------
        // 3. Read YOLOv8 NMS output
        // ---------------------------------------------------------

        println!("OCR: About to read output");

        let buf_size = self.ocr_output_sizes[0];
        let mut output_buf = vec![0u8; buf_size];

        unsafe {
            let status = hailo_vstream_read_raw_buffer(
                self.ocr_output_vstreams[0],
                output_buf.as_mut_ptr() as *mut std::ffi::c_void,
                buf_size,
            );

            println!("OCR: Output read: {}", status);

            if status != HAILO_SUCCESS {
                return None;
            }
        }

        // ---------------------------------------------------------
        // 4. Decode Hailo NMS_BY_CLASS output
        //
        // Each class contains:
        //
        //     float32 bbox_count
        //
        // followed by bbox_count bounding boxes:
        //
        //     y_min
        //     x_min
        //     y_max
        //     x_max
        //     score
        //
        // Each value is float32.
        // ---------------------------------------------------------

        const NUM_CLASSES: usize = 36;
        const BYTES_PER_BBOX: usize = 20;

        // Character mapping:
        //
        //  0-9 -> digits
        // 10-35 -> A-Z
        //
        // This assumes the YOLOv8-LP model was trained with
        // this class ordering.
        const CHAR_MAP: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        #[derive(Debug)]
        struct CharacterDetection {
            class_id: usize,
            x_min: f32,
            score: f32,
        }

        let mut offset = 0usize;
        let mut characters = Vec::<CharacterDetection>::new();

        for class_id in 0..NUM_CLASSES {
            // Need four bytes for bbox_count.
            if offset + 4 > output_buf.len() {
                break;
            }

            let bbox_count = f32::from_le_bytes([
                output_buf[offset],
                output_buf[offset + 1],
                output_buf[offset + 2],
                output_buf[offset + 3],
            ]) as usize;

            offset += 4;

            for _ in 0..bbox_count {
                if offset + BYTES_PER_BBOX > output_buf.len() {
                    println!(
                        "OCR: NMS buffer ended at class {}, offset {}",
                        class_id, offset
                    );
                    return None;
                }

                let _y_min = f32::from_le_bytes([
                    output_buf[offset],
                    output_buf[offset + 1],
                    output_buf[offset + 2],
                    output_buf[offset + 3],
                ]);

                let x_min = f32::from_le_bytes([
                    output_buf[offset + 4],
                    output_buf[offset + 5],
                    output_buf[offset + 6],
                    output_buf[offset + 7],
                ]);

                let _y_max = f32::from_le_bytes([
                    output_buf[offset + 8],
                    output_buf[offset + 9],
                    output_buf[offset + 10],
                    output_buf[offset + 11],
                ]);

                let _x_max = f32::from_le_bytes([
                    output_buf[offset + 12],
                    output_buf[offset + 13],
                    output_buf[offset + 14],
                    output_buf[offset + 15],
                ]);

                let score = f32::from_le_bytes([
                    output_buf[offset + 16],
                    output_buf[offset + 17],
                    output_buf[offset + 18],
                    output_buf[offset + 19],
                ]);

                offset += BYTES_PER_BBOX;

                characters.push(CharacterDetection {
                    class_id,
                    x_min,
                    score,
                });
            }
        }

        println!("OCR: detected {} characters", characters.len());

        if characters.is_empty() {
            return None;
        }

        // ---------------------------------------------------------
        // 5. Sort characters from left to right
        // ---------------------------------------------------------

        characters.sort_by(|a, b| {
            a.x_min.total_cmp(&b.x_min)
        });

        // ---------------------------------------------------------
        // 6. Convert class IDs to characters
        // ---------------------------------------------------------

        let mut plate_text = String::new();

        for character in characters {
            if character.class_id < CHAR_MAP.len() {
                let ch = CHAR_MAP[character.class_id] as char;

                println!(
                    "OCR: character '{}' class={} x={:.3} score={:.3}",
                    ch,
                    character.class_id,
                    character.x_min,
                    character.score
                );

                plate_text.push(ch);
            }
        }

        if plate_text.is_empty() {
            None
        } else {
            println!("OCR: Plate text = {}", plate_text);
            Some(plate_text)
        }
    }
}