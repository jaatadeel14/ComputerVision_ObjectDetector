# Real-Time Computer Vision: Object Detection & Shape Analysis

A real-time computer vision application built with **Python, OpenCV, and YOLO11n**. The system detects and classifies objects from a webcam feed while applying traditional image processing techniques to extract shapes and analyze their geometric properties.

## Features

- Real-time object detection using the pretrained YOLO11n model.
- Gaussian blur for image smoothing and noise reduction.
- Grayscale conversion and adaptive thresholding.
- Contour extraction with configurable pixel-area filtering.
- Dynamic class-wise object counting.
- Live display of estimated FPS and processing time.
- CSV-based performance logging.
- Performance analysis with automatically generated charts.
- Detailed computer vision whitepaper.

## Technologies

- Python
- OpenCV
- Ultralytics YOLO11n
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text
ComputerVision_ObjectDetector/
├── main.py
├── analyze_performance.py
├── analyze_results.py
├── requirements.txt
├── README.md
├── .gitignore
├── outputs/
│   ├── performance_log.csv
│   ├── performance_summary.txt
│   ├── processing_time.png
│   ├── processing_fps.png
│   ├── fps_analysis.png
│   └── object_counts.png
└── reports/
    └── Computer_Vision_Whitepaper.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jaatadeel14/ComputerVision_ObjectDetector.git
cd ComputerVision_ObjectDetector
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Run the real-time detection pipeline

```bash
python main.py
```

The application displays the object detection output, adaptive threshold image, and contour extraction results.

Press **Q** while a video window is focused to stop the application.

### Analyze performance

```bash
python analyze_performance.py
```

The script summarizes recorded metrics and generates performance charts in the `outputs/` directory.

## Performance Evaluation

An initial webcam experiment recorded the following results:

| Metric | Result |
|---|---:|
| Recorded frames | 294 |
| Average processing time | 78.12 ms |
| Average estimated FPS | 15.15 |
| Average detections per frame | 1.02 |
| Average extracted shapes per frame | 1.55 |

These measurements describe the recorded test run, not a guarantee of performance on other hardware or environments. The reported FPS is an estimate derived from processing time.

## Documentation

The detailed project whitepaper is available at:

[`reports/Computer_Vision_Whitepaper.md`](reports/Computer_Vision_Whitepaper.md)

## Limitations

- Detection quality depends on the pretrained model and its supported classes.
- Lighting, object size, occlusion, and background complexity can affect results.
- Contour counts do not necessarily correspond to actual object counts.
- The current implementation counts detections independently in each frame and does not track