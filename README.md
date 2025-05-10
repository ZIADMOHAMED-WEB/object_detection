# object_detection

This project combines **YOLOv8 object detection** with **stereo depth estimation** using stereo images from the **KITTI dataset**. It computes the distance to detected objects in real-world meters by estimating disparity from stereo image pairs and visualizes both the detection and the depth.

---

## 🔍 Features

- Detects objects using [YOLOv8](https://github.com/ultralytics/ultralytics)
- Computes depth maps from stereo image pairs using OpenCV
- Overlays bounding boxes with labels and real-world depth
- Visualizes the annotated output with optional FPS display

---

## 📂 Folder Structure

project-root/
│
├── data/
│ └── 2011_09_26_drive_0001_sync/
│ ├── image_02/data/ ← Left camera images
│ └── image_03/data/ ← Right camera images
│
├── yolov8_depth_estimation.py
└── README.md

yaml
Copy
Edit

---

## 🧰 Requirements

Install the required Python libraries:

```bash
pip install ultralytics opencv-python numpy
🔧 Setup
Download a sequence from the KITTI Vision Benchmark Suite.

Extract it and place the stereo image folders in the data/ directory.

Left images path: image_02/data

Right images path: image_03/data

Download a YOLOv8 model (e.g., yolov8n.pt) from Ultralytics and ensure it's accessible in the script.

▶️ Running the Code
bash
Copy
Edit
python yolov8_depth_estimation.py
Press Q to quit the visualization window.

📸 Output
Bounding boxes with class name, confidence, and estimated distance (in meters)

Optionally show a color-mapped depth map and FPS

⚙️ Technical Details
Depth Estimation: depth = (focal_length * baseline) / disparity

Focal length and baseline are hardcoded from KITTI calibration

Disparity Calculation: Using StereoBM or optionally StereoSGBM

Object Detection: YOLOv8 via ultralytics Python package

🧠 Notes
Disparity values ≤ 0 are clamped to avoid division errors.

StereoBM may produce noisy results; tuning parameters or using SGBM may improve depth accuracy.

Only the center pixel of each bounding box is used for depth estimation.

📄 License
MIT License. Use at your own risk.
