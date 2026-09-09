// ✅✅✅ MULTIPLE LINE CHANGES BELOW ✅✅✅
// ✅ New file for the Axum web server and route handlers
use axum::{
    extract::State,
    routing::{post, get},
    Json, Router,
    response::{IntoResponse, Response, Html},
    body::Body,
    http::header,
};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tokio::sync::{watch, broadcast};
use std::sync::Arc;

use crate::config::{self, AppConfig};

// ✅ AppState moved here and made public so `main.rs` can initialize it
#[derive(Clone)]
pub struct AppState {
    pub config_tx: Arc<watch::Sender<AppConfig>>,
    pub stream_tx: broadcast::Sender<Vec<u8>>,
}

// ✅ Encapsulated router setup and server binding
pub async fn run_server(state: AppState) -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new()
        .route("/", get(index_page))
        .route("/api/config", post(update_config))
        .route("/stream", get(mjpeg_stream_handler))
        .with_state(state);

    println!("Dashboard running at http://0.0.0.0:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn update_config(
    State(state): State<AppState>,
    Json(new_config): Json<AppConfig>,
) -> Json<AppConfig> {
    if let Err(e) = config::save_config(&new_config) {
        eprintln!("Failed to save configuration to disk: {:?}", e);
    } else {
        println!("Configuration successfully written to config.json");
    }

    let _ = state.config_tx.send(new_config.clone());

    Json(new_config)
}

// ✅ New function to serve the HTML dashboard
async fn index_page() -> Html<&'static str> {
    Html(r#"
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Counting Cars Dashboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #121212;
                    color: #ffffff;
                    text-align: center;
                    margin: 0;
                    padding: 20px;
                }
                h1 {
                    color: #00ffcc;
                    margin-bottom: 20px;
                }
                .video-wrapper {
                    max-width: 1280px;
                    margin: 0 auto;
                    border: 4px solid #333;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.8);
                }
                img {
                    width: 100%;
                    height: auto;
                    display: block; /* Removes bottom margin gap */
                }
                .status {
                    margin-top: 15px;
                    font-size: 0.9em;
                    color: #aaaaaa;
                }
            </style>
        </head>
        <body>
            <h1>Gate Surveillance - Camera 0</h1>
            <div class="video-wrapper">
                <img src="/stream" alt="Live Gate Feed" />
            </div>
            <div class="status">
                🟢 System Online | Awaiting ALPR Triggers
            </div>
        </body>
        </html>
    "#)
}

async fn mjpeg_stream_handler(State(state): State<AppState>) -> impl IntoResponse {
    let rx = state.stream_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| {
        match msg {
            Ok(data) => {
                let header = format!("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", data.len());
                let mut packet = header.into_bytes();
                packet.extend(data);
                packet.extend_from_slice(b"\r\n");
                Some(Ok::<_, std::io::Error>(packet))
            }
            Err(_) => None,
        }
    });
    Response ::builder()
        .header(header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame")
        .body(Body::from_stream(stream))
        .unwrap()
}