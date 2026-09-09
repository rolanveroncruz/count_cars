/*
The configuration system stores AppConfig as the configuration.
- get_or_create_initial_config() loads the config from config.json or creates a default one if it doesn't exist.
- save_config() serializes and writes the configuration to config.json
 */
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub min_box_area: f64, //  1. Minimum bounding box area to start tracking an object.
    pub confidence: f32,   //  2. YOLO confidence threshold for vehicle detection
    pub alpr_trigger_area: f64, // 2. Minimum bounding box area to trigger ALPR/OCR capture
    pub gate_line_y: i32,   // 4. Y-coordinate pixel line across for gate crossing detection.
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            min_box_area: 3000.0,
            confidence: 0.45,
            alpr_trigger_area: 25000.0,
            gate_line_y: 600,
        }
    }
}

/// Loads config from config.json or creates a default one if it doesn't exist.
pub fn get_or_create_initial_config() -> AppConfig {
    let path = "config.json";
    if Path::new(path).exists() {
        match fs::read_to_string(path) {
            Ok(content) => {
                if let Ok(config) = serde_json::from_str(&content) {
                    return config;
                }
            }
            Err(_) => eprintln!("Failed to read config.json, using defaults."),
        }
    }

    let default_config = AppConfig::default();
    let _ = save_config(&default_config);
    default_config
}

/// Serializes and writes the configuration to config.json
pub fn save_config(config: &AppConfig) -> Result<(), std::io::Error> {
    let path = "config.json";
    let json = serde_json::to_string_pretty(config)?;
    fs::write(path, json)?;
    Ok(())
}