# CATD

## Overview
CATD is a computer vision project that utilizes YOLOv8 for object detection and video analysis. The repository includes scripts for running detection pipelines, extracting frames, logging, and viewing results.

## Features
- Object detection using YOLOv8
- Video processing and frame extraction
- Logging and result visualization
- Modular pipeline for easy extension

## Project Structure
- `app.py`: Main application entry point
- `pipeline.py`: Detection pipeline logic
- `extract_top_frames.py`: Extracts top frames from videos
- `viewer.py`: Visualization and result viewing
- `logger.py`: Logging utilities
- `yolov8n.pt`: YOLOv8 model weights
- `detections/`: Output detection results
- `Videos/`: Input video files
- `requirements.txt`: Python dependencies

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Agrim-Sigdel/CATD.git
   cd CATD
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Run the main application:
  ```bash
  python app.py
  ```
- Extract top frames:
  ```bash
  python extract_top_frames.py
  ```
- View results:
  ```bash
  python viewer.py
  ```

## Requirements
See `requirements.txt` for the full list of dependencies.


## Environment Variables
You need to provide Roboflow API information for the project to work. Create a `.env` file in the root directory (see `.env.example` for reference) and add the following:

```
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_WORKSPACE=your_roboflow_workspace
ROBOFLOW_PROJECT=your_roboflow_project
```

## Notes
- Place your input videos in the `Videos/` directory.
- Detection results will be saved in the `detections/` directory.
- The YOLOv8 model weights file (`yolov8n.pt`) should be present in the root directory.


