### Architectural Decisions & Findings (2026-09-09)

#### 1. Camera Input Strategy
- **Decision:** Continue using `rpicam-vid` piped into Rust via `stdout` with `--codec mjpeg`.
- **Reasoning:** 
    - **Simplicity:** Avoids complex GStreamer pipelines or C++ FFI.
    - **Parsing:** Easy manual frame delimiting using JPEG start/end markers (`0xFF, 0xD8` and `0xFF, 0xD9`).
    - **Performance:** While RPi 5 lacks a hardware JPEG decoder, its Cortex-A76 cores handle software decoding of 640x640 @ 30fps with minimal overhead (<15% single-core load).

#### 2. ALPR Pipeline (Subdivision Gate Use Case)
- **Use Case:** Single-vehicle entry/exit (serial processing).
- **Strategy:** Sequential "Detect -> Crop -> Detect -> OCR" pipeline.
- **Hardware Acceleration:** Hailo-8 (26 TOPS) will be used for all three stages (Vehicle, Plate, OCR).
- **Implementation:**
    - Models will be loaded into the Hailo-8 using **Context Switching**.
    - Switching time is <1ms, making it feasible within the 33ms window of a 30fps stream.
    - CPU will handle image cropping via OpenCV between model passes.

#### 3. Bottleneck Analysis (RPi 5)
- **Low Risk:** MJPEG decoding and sequential AI inference.
- **Moderate Risk:** JPEG re-encoding for the web stream/archiver. (Keep quality at 75% or lower).
- **Optimization Path:** If latency increases, switch pipe to `--codec yuv420` to eliminate JPEG decoding overhead entirely.
