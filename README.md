# Overview
This project is a rust project for recording vehicles entering and exiting a subdivision gate.
The main challenge is recording the license plate of vehicles which may be 4-wheeled vehicles like cars, trucks, or vans,
or two-wheeled vehicles like motorcycles which have their license plates installed on their rear ends.

## Hardware and Strategy
The software is written in rust and will run on a Raspberry Pi 5 with an AI HAT+ hailo8 (26TOPS).
It will have two cameras: one raspberry pi camera module 3 to detect and capture license plates of approaching 4-wheeled vehicles,
and a rear faceing camera to detect and capture license plates of approaching 2-wheeled vehicles, or vehicles whose license plates  
were not captured by the front-facing camera.


## Process
The process is to use the front-facing camera to:
1. First detect an oncoming vehicle using YOLO. If the vehicle is close enough, meaning its area in the frame is greater than a certain threshold,
2. to crop the vehicle from the frame, and then use the hailo8 to detect the license plate, also using YOLO.
3. If the license plate is detected, then capture the image of the license plate and use an OCR to read the license plate. 
4. We then record this to a sqlite database. 

We then monitor the rear-facing camera's input stream:
1. We assume the vehicle that is visible is the same vehicle that was detected by the front-facing camera.
2. We monitor camera frames, attempting to detect the license plate of the vehicle using the same process as the front-facing camera.
3. If the license plate is detected, then capture the image of the license plate and use an OCR to read the license plate.The license plate
is then recorded to the same sqlite table row as the assumed front-facing camera vehicle.

## Models
1. Vehicle Detection: Yolo26n is the latest cutting edge model from Ultralytics 
2. License plate detecion - yolov8n_license_plates
3. License plate ocr - lprnet