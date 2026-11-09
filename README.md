# SnakeWatch

Real-Time Snake Intrusion Detection and Human Safety Monitoring.

## Main components
- Fine-tuned RT-DETRv2 snake detection.
- ByteTrack multi-object tracking.
- Boundary simulation and movement-state analysis.
- Evidence video and event generation.
- Human Safety Monitoring.
- Dataset preparation and model evaluation.

## Repository structure
- app/: Streamlit dashboard and SnakeWatch modules.
- src/: Dataset preparation, detection, tracking and evaluation.
- documentation/: Supporting documentation.

## Model
Uses RT-DETRv2 based on PekingU/rtdetr_v2_r18vd.
Model weights are stored separately and are not included.

## Dataset
Datasets and input videos are stored separately and are not included.
Configure local paths before running the scripts.

## Installation
Install the dependencies required by the project environment.

## Run dashboard
streamlit run app/snakewatch_speed_test.py

## Limitations
- Research prototype, not a certified safety system.
- Boundary zones use image coordinates, not calibrated distances.

Generated videos, datasets, checkpoints, logs and backups are excluded.