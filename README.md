Automatic License Plate Recognition (ALPR) – Final Project
📌 Project Overview

This project implements an end-to-end Automatic License Plate Recognition (ALPR) system using Computer Vision and Deep Learning.
The pipeline detects license plates from video streams, tracks them across frames, recognizes alphanumeric characters using OCR, and evaluates performance using real-time metrics.

The project is structured as a Proof of Concept (POC) suitable for academic evaluation and real-world deployment scenarios.

Folder Structure
    ALPR final/
    │
    ├── test videos/
    │   └── (Input videos used for testing the ALPR pipeline)
    │
    ├── code files/
    │   └── (Python scripts implementing detection, tracking, OCR, and evaluation)
    │
    ├── flowchart.drawio.png
    │   └── System architecture / methodology flowchart
    │
    ├── output_with_metrics.mp4
    │   └── Final annotated output video with FPS and latency overlays
    │
    ├── output_with_license_BOXLOCKED.mp4
    │   └── Output video with locked bounding boxes and stabilized OCR text
    │
    ├── performance_log.csv
    │   └── Performance metrics log (FPS, detection latency, OCR latency, total latency)
    │
    ├── runs.zip
    │   └── YOLOv8 training runs (weights, logs, evaluation metrics)
    │
    ├── License_Plate_Recognition.v1i.yolov8.zip
    │   └── Dataset used for training and evaluation (YOLOv8 format)
    │
    └── README.md

Performance Evaluation

    Performance metrics are stored in: performance_log.csv

    Each row contains:

    Frame number

    FPS

    Detection latency (ms)

    OCR latency (ms)

    Total end-to-end latency (ms)

    These metrics were used to generate FPS and latency graphs for analysis.

🎥 Output Files

    output_with_metrics.mp4
    Displays bounding boxes, recognized plate text, FPS, and latency overlays.

    output_with_license_BOXLOCKED.mp4
    Shows stabilized license plate text with locked bounding boxes and confidence-based color coding.

📂 Dataset

    License_Plate_Recognition.v1i.yolov8.zip

    Annotated license plate images

    YOLOv8-compatible format

    Train / Validation / Test splits included

▶️ How to Run

    Use license_plate_best.pt model 
    
    Place input videos inside:

        test videos/


    Run the Python scripts inside:

        code files/


    Outputs will be generated as:

        Annotated videos (.mp4)

        Performance logs (.csv)