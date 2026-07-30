# Stamp Detection System

A computer vision system for detecting stamps on inspection check sheets.

The system uses Hough Circle Detection as the primary detection method and YOLOv8 as a fallback for cases where the stamp is faint or difficult to detect using traditional image processing.

## Detection Pipeline

Check Sheet
    |
    v
Region Extraction
    |
    v
Hough Circle Detection
    |
    +---- Detected ----> OK
    |
    v
YOLOv8
    |
    +---- Detected ----> OK
    |
    v
NG

## Features

- Region-based stamp detection
- Hough Circle Detection
- YOLOv8 fallback detection
- Detection of faint stamps
- CSV result logging
- Debug image generation
- GPU support for YOLO training

## Project Structure

stamp-detection/

    stamp_detector.py
    faint_stamp_detector.py
    train_yolo.py
    prepare_yolo_dataset.py
    stamp_data.yaml
    requirements.txt
    .gitignore
    README.md

    train/
        blank_stamps/
        faint_stamps/

    datasets/
        stamps/

    debug_regions/

    runs/

## Dataset

The training dataset contains two categories:

- faint_stamps
- blank_stamps

The images are already cropped to the stamp regions.

The dataset preparation script converts the images into YOLO format.

Run:

python prepare_yolo_dataset.py

## Training

After preparing the dataset, train the YOLO model using:

python train_yolo.py

The trained model is saved at:

runs/detect/yolov8_stamp/weights/best.pt

## Testing YOLO

The YOLO model can be tested separately using:

python faint_stamp_detector.py

## Running the Complete Detector

Update the checksheet image path in stamp_detector.py:

CHECKSHEET_PATH = "20250615.jpg"

Then run:

python stamp_detector.py

The detector first attempts Hough Circle Detection.

If Hough does not detect a stamp, the cropped region is passed to YOLOv8.

## Output

The system generates:

output_checksheet_with_boxes.jpg

stamp_detection_results.csv

Debug images are stored in:

debug_regions/

## Detection Regions

The current system processes six regions:

- exterior_lh
- exterior_rh
- exterior_final
- interior_lh
- interior_rh
- interior_final

The coordinates can be modified in stamp_detector.py.

## Technologies

- Python
- OpenCV
- NumPy
- Pandas
- PyTorch
- Ultralytics YOLOv8
- Hough Circle Transform

## Installation

Install the required packages:

pip install -r requirements.txt

## GPU

If CUDA-enabled PyTorch is installed and an NVIDIA GPU is available, YOLO training automatically uses the GPU.

GPU availability can be checked using:

python -c "import torch; print(torch.cuda.is_available())"
