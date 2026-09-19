use opencv::{
    core::{Mat,Size},
    imgproc,
    prelude::*,
};

use crate::inference::bindings::{hailo_vstream_write_raw_buffer, HAILO_SUCCESS, hailo_vstream_read_raw_buffer};
use crate::inference::CountCarsIntelligence;
use crate::inference::hailo8::Hailo8Manager;

/// A single license-plate detection produced by the Tiny YOLOv4 decoder.
///
/// Coordinates are in the coordinate system of the 416x416 input image.
#[derive(Debug, Clone)]
pub struct PlateDetection {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub score: f32,
}

/// Decode one Tiny YOLOv4 output tensor.
///
/// The Hailo HEF gives us the tensor in host-side NHWC order:
///
///     [height][width][18 channels]
///
/// There are 3 anchors at every grid cell, and each anchor has
/// 6 values:
///
///     x, y, width, height, objectness, class_score
///
/// Therefore:
///
///     3 anchors × 6 values = 18 channels
///
/// For example, at grid cell (x=4, y=7):
///
///     channels  0.. 5  -> anchor 0
///     channels  6..11  -> anchor 1
///     channels 12..17  -> anchor 2
///
/// `output` is already FLOAT32 because our Hailo output vstream
/// requests HAILO_FORMAT_TYPE_FLOAT32.
///
/// `grid_w` and `grid_h` are normally:
///
///     stride 32 -> 13 × 13
///     stride 16 -> 26 × 26
///
/// `anchors` contains three (width, height) anchor pairs for this
/// particular output layer.
///
/// `stride` tells us how large one grid cell is in the original
/// 416 × 416 image.
///
/// `score_threshold` removes very weak detections.
pub fn decode_plate_output(
    output: &[f32],
    grid_w: usize,
    grid_h: usize,
    anchors: &[(f32, f32)],
    stride: f32,
    score_threshold: f32,
) -> Vec<PlateDetection> {
    // Tiny YOLOv4 has exactly 3 anchors per grid cell.
    debug_assert_eq!(anchors.len(), 3);

    // Each anchor occupies six channels:
    //
    //     0 = x
    //     1 = y
    //     2 = width
    //     3 = height
    //     4 = objectness
    //     5 = class score
    //
    // With three anchors this gives 18 channels total.
    const VALUES_PER_ANCHOR: usize = 6;
    const CHANNELS: usize = 18;

    debug_assert_eq!(
        output.len(),
        grid_w * grid_h * CHANNELS,
        "Unexpected YOLO output size"
    );

    let mut detections = Vec::new();

    // YOLO evaluates every position in the output grid.
    //
    // For the 13x13 output there are:
    //
    //     13 × 13 = 169 cells
    //
    // For the 26x26 output there are:
    //
    //     26 × 26 = 676 cells
    //
    for grid_y in 0..grid_h {
        for grid_x in 0..grid_w {
            // Hailo has given us the tensor in host-side NHWC order.
            //
            // Therefore the first value for this grid cell is:
            //
            //     (grid_y * grid_w + grid_x) * 18
            //
            // Think of the output as:
            //
            //     cell 0: [18 values]
            //     cell 1: [18 values]
            //     cell 2: [18 values]
            //     ...
            //
            // and each cell's 18 values contain the three anchors.
            let cell_base = (grid_y * grid_w + grid_x) * CHANNELS;

            for anchor_index in 0..anchors.len() {
                // Each anchor gets six consecutive channels.
                //
                // Anchor 0:
                //     0, 1, 2, 3, 4, 5
                //
                // Anchor 1:
                //     6, 7, 8, 9, 10, 11
                //
                // Anchor 2:
                //     12, 13, 14, 15, 16, 17
                let base = cell_base + anchor_index * VALUES_PER_ANCHOR;

                let raw_x = output[base];
                let raw_y = output[base + 1];
                let raw_w = output[base + 2];
                let raw_h = output[base + 3];
                let raw_objectness = output[base + 4];
                let raw_class = output[base + 5];

                // ---------------------------------------------------------
                // YOLOv4 decoding
                // ---------------------------------------------------------
                //
                // The network does NOT directly output:
                //
                //     center_x
                //     center_y
                //     width
                //     height
                //
                // Instead it outputs "raw" values which need to be
                // transformed.
                //
                // The Model Zoo's _yolo4_decode() does:
                //
                //     box_scales = exp(raw_box_scales) * anchor
                //
                //     box_centers =
                //         (sigmoid(raw_box_centers) * scale_x_y
                //          - 0.5 * (scale_x_y - 1)
                //          + grid_offset)
                //         * stride
                //
                // Tiny YOLOv4 uses scale_x_y = 1.05.
                const SCALE_X_Y: f32 = 1.05;

                // Convert the raw x/y network outputs into values
                // relative to this particular grid cell.
                //
                // sigmoid() turns an arbitrary number into a value
                // between 0 and 1.
                //
                // grid_x/grid_y then tells us WHICH cell we're in.
                let decoded_x = (
                    sigmoid(raw_x) * SCALE_X_Y
                        - 0.5 * (SCALE_X_Y - 1.0)
                        + grid_x as f32
                ) * stride;

                let decoded_y = (
                    sigmoid(raw_y) * SCALE_X_Y
                        - 0.5 * (SCALE_X_Y - 1.0)
                        + grid_y as f32
                ) * stride;

                // Width and height work differently.
                //
                // The network predicts a logarithmic scale.
                // exp() converts that back to a multiplicative scale.
                //
                // Then we multiply by the anchor dimensions.
                let (anchor_w, anchor_h) = anchors[anchor_index];

                let decoded_w = raw_w.exp() * anchor_w;
                let decoded_h = raw_h.exp() * anchor_h;

                // Objectness answers:
                //
                //     "Does this anchor contain an object?"
                //
                // The class prediction answers:
                //
                //     "How likely is this object to be a license plate?"
                //
                // Both are raw network outputs, so both go through
                // sigmoid().
                let objectness = sigmoid(raw_objectness);
                let class_score = sigmoid(raw_class);

                // The Model Zoo combines them by multiplication:
                //
                //     final_score = objectness × class_score
                //
                // Since this model has only ONE class, this is simply
                // the confidence that this particular prediction is
                // a license plate.
                let score = objectness * class_score;

                // Ignore weak predictions.
                //
                // The Model Zoo YAML specifies:
                //
                //     score_threshold: 0.1
                //
                if score < score_threshold {
                    continue;
                }

                // We currently have:
                //
                //     center = (decoded_x, decoded_y)
                //     size   = (decoded_w, decoded_h)
                //
                // Convert center/size into corner coordinates.
                //
                //              width
                //        <---------------->
                //
                //        xmin              xmax
                //          |----------------|
                //          |                |
                //          |       +        |  <- center
                //          |                |
                //          |----------------|
                //        ymin              ymax
                //
                let half_w = decoded_w / 2.0;
                let half_h = decoded_h / 2.0;

                let xmin = decoded_x - half_w;
                let ymin = decoded_y - half_h;
                let xmax = decoded_x + half_w;
                let ymax = decoded_y + half_h;

                detections.push(PlateDetection {
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    score,
                });
            }
        }
    }

    detections
}

/// YOLO uses the sigmoid function for x/y, objectness and class scores.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Hailo8Manager{
    pub fn detect_plate(&mut self, frame: &Mat) -> Vec<PlateDetection> {
        let mut rgb_frame = Mat::default();

        imgproc::cvt_color_def(
            frame,
            &mut rgb_frame,
            opencv::imgproc::COLOR_BGR2RGB,
        );

        let mut resized = Mat::default();

        imgproc::resize(
            &rgb_frame,
            &mut resized,
            Size::new(416,416),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        );

        // Convert the image into the contiguous RGB byte buffer
        // expected by the Hailo input vstream.
        let frame_data = resized
            .data_bytes()
            .expect("Failed to get resized frame bytes");

        assert_eq!(
            frame_data.len(),
            self.plate_input_size,
            "Plate input size mismatch: got {}, expected {}",
            frame_data.len(),
            self.plate_input_size
        );

        // Send the image to the Tiny YOLOv4 network.
        let status = unsafe {
            hailo_vstream_write_raw_buffer(
                self.plate_input_vstream,
                frame_data.as_ptr() as *const std::ffi::c_void,
                frame_data.len(),
            )
        };

        if status != HAILO_SUCCESS {
            panic!("Failed to write plate input: {}", status);
        }

        // We have TWO YOLO output tensors:
        //
        //   output 0 -> 13 x 13 x 18, stride 32
        //   output 1 -> 26 x 26 x 18, stride 16
        //
        // Each grid cell contains predictions for three anchors.
        //
        // Each anchor contains:
        //
        //   x, y, width, height, objectness, class
        //
        // Therefore:
        //
        //   3 anchors × 6 values = 18 channels.
        //
        // We'll decode both tensors and combine their detections.
        let mut all_detections = Vec::new();

        for i in 0..self.plate_output_vstreams.len() {
            let output_size = self.plate_output_sizes[i];

            // HailoRT will give us FLOAT32 because the output
            // vstreams were created with HAILO_FORMAT_TYPE_FLOAT32.
            let mut output_buf = vec![0u8; output_size];

            let status = unsafe {
                hailo_vstream_read_raw_buffer(
                    self.plate_output_vstreams[i],
                    output_buf.as_mut_ptr() as *mut std::ffi::c_void,
                    output_buf.len(),
                )
            };

            if status != HAILO_SUCCESS {
                panic!("Failed to read plate output {}", i);
            }

            let float_count = output_buf.len() / std::mem::size_of::<f32>();

            let output_floats = unsafe {
                std::slice::from_raw_parts(
                    output_buf.as_ptr() as *const f32,
                    float_count,
                )
            };

            // ---------------------------------------------------------
            // Select the parameters belonging to this output tensor.
            // ---------------------------------------------------------
            //
            // Output 0:
            //
            //     13 x 13 grid
            //     stride 32
            //     large anchors
            //
            // Output 1:
            //
            //     26 x 26 grid
            //     stride 16
            //     small anchors
            //
            let (grid_w, grid_h, stride, anchors) = if i == 0 {
                (
                    13,
                    13,
                    32.0,
                    [
                        (81.0, 82.0),
                        (135.0, 169.0),
                        (344.0, 319.0),
                    ],
                )
            } else {
                (
                    26,
                    26,
                    16.0,
                    [
                        (10.0, 14.0),
                        (23.0, 27.0),
                        (37.0, 58.0),
                    ],
                )
            };

            // ---------------------------------------------------------
            // Decode the raw YOLO tensor.
            // ---------------------------------------------------------
            //
            // The decoder:
            //
            //   raw x/y -> sigmoid + grid position
            //   raw w/h -> exp + anchor dimensions
            //   objectness -> sigmoid
            //   class -> sigmoid
            //   score = objectness * class
            //
            // It also converts center/width/height into:
            //
            //   xmin, ymin, xmax, ymax
            //
            // and discards predictions below the 0.1 threshold.
            let detections = decode_plate_output(
                output_floats,
                grid_w,
                grid_h,
                &anchors,
                stride,
                0.1,
            );

            println!(
                "Plate Output {}: {} detections above threshold",
                i,
                detections.len()
            );

            all_detections.extend(detections);
        }

        println!(
            "Total plate detections before NMS: {}",
            all_detections.len()
        );
        let all_detections = non_max_suppression(
            all_detections,
            0.3,
        );

        println!(
            "Total plate detections after NMS: {}",
            all_detections.len()
        );

        for detection in &all_detections {
            println!(
                "  plate: ({:.1}, {:.1}) - ({:.1}, {:.1}), score={:.4}",
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
                detection.score,
            );
        }

        // NMS will be added next.
        //
        // For now we return every detection that survived
        // the score threshold.
        all_detections
    }
}
/// Calculate Intersection over Union (IoU) between two bounding boxes.
///
/// IoU measures how much two boxes overlap:
///
///     IoU = intersection_area / union_area
///
/// A value of:
///     0.0 -> no overlap
///     1.0 -> identical boxes
fn calculate_iou(a: &PlateDetection, b: &PlateDetection) -> f32 {
    // Find the coordinates of the intersection rectangle.
    let intersection_xmin = a.xmin.max(b.xmin);
    let intersection_ymin = a.ymin.max(b.ymin);
    let intersection_xmax = a.xmax.min(b.xmax);
    let intersection_ymax = a.ymax.min(b.ymax);

    // If the boxes don't overlap, intersection area is zero.
    let intersection_width = (intersection_xmax - intersection_xmin).max(0.0);
    let intersection_height = (intersection_ymax - intersection_ymin).max(0.0);

    let intersection_area = intersection_width * intersection_height;

    // Calculate each box's area.
    let area_a = (a.xmax - a.xmin).max(0.0)
        * (a.ymax - a.ymin).max(0.0);

    let area_b = (b.xmax - b.xmin).max(0.0)
        * (b.ymax - b.ymin).max(0.0);

    // The union is everything covered by either box.
    let union_area = area_a + area_b - intersection_area;

    if union_area <= 0.0 {
        return 0.0;
    }

    intersection_area / union_area
}
/// Non-Maximum Suppression (NMS).
///
/// Keeps the highest-confidence detection and suppresses
/// overlapping detections whose IoU is above the threshold.
///
/// For our Tiny YOLOv4 plate model, the YAML specifies:
///
///     nms_iou_thresh = 0.3
fn non_max_suppression(
    mut detections: Vec<PlateDetection>,
    iou_threshold: f32,
) -> Vec<PlateDetection> {
    // Highest-confidence detections must be considered first.
    detections.sort_by(|a, b| {
        b.score.total_cmp(&a.score)
    });

    let mut kept = Vec::new();

    while let Some(best) = detections.first().cloned() {
        // The strongest remaining detection survives.
        kept.push(best.clone());

        // Remove it from the remaining candidates.
        detections.remove(0);

        // Suppress detections that overlap the selected
        // detection too much.
        detections.retain(|candidate| {
            let iou = calculate_iou(&best, candidate);

            // Keep the candidate only if its overlap is
            // below the NMS threshold.
            iou <= iou_threshold
        });
    }

    kept
}

