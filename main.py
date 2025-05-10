import cv2
import numpy as np
import os
from ultralytics import YOLO

# === Helper Functions ===

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

# === Paths ===
left_folder = 'data/2011_09_26_drive_0001_sync/image_02/data'
right_folder = 'data/2011_09_26_drive_0001_sync/image_03/data'

# === Load Images ===
left_images = load_images_from_folder(left_folder)
right_images = load_images_from_folder(right_folder)

print(f"Loaded {len(left_images)} left images and {len(right_images)} right images.")

# === Load YOLO Model ===
model = YOLO('yolov8n.pt')  # small model, you can change to yolov8s.pt etc.

# === Create StereoBM matcher for depth ===
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# KITTI Camera Calibration
focal_length = 721.0  # pixels
baseline = 0.54       # meters

# === Processing Loop ===
for idx in range(len(left_images)):
    print(f"Processing Frame {idx}")

    left_img = left_images[idx]
    right_img = right_images[idx]

    # === Compute Disparity ===
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Avoid division by zero
    disparity[disparity <= 0.0] = 0.1

    # === Compute Depth Map ===
    depth_map = (focal_length * baseline) / disparity

    # === Run YOLO Detection ===
    results = model.predict(source=left_img, imgsz=640, conf=0.5, verbose=False)[0]

    annotated_img = left_img.copy()

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            cls_id = int(box.cls[0])

            # Center of the box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Get depth value at center
            if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
                object_depth = depth_map[center_y, center_x]
            else:
                object_depth = -1  # invalid

            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{model.names[cls_id]} {conf:.2f} | {object_depth:.2f}m"
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # === Show Results ===
    cv2.imshow("YOLO + Depth", annotated_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
