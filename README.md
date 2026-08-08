# AI-Driven Vehicle Number Plate Recognition System

An end-to-end Computer Vision pipeline that automates vehicle license plate detection and Optical Character Recognition (OCR). This system utilizes custom-trained/pretrained **YOLOv5** for high-accuracy bounding box detection of license plates, combined with **EasyOCR** for text extraction from images and video streams.

---

## 📌 Features

* **Real-time & Batch Processing**: Supports detection and recognition on both static images and video feeds.
* **Deep Learning Object Detection**: Powered by **YOLOv5** to accurately locate license plates under varying lighting and structural angles.
* **Text Extraction (OCR)**: Integrates **EasyOCR** for robust string extraction from detected Regions of Interest (ROI).
* **Automated Visual Annotation**: Draws bounding boxes around license plates and overlays detected plate text directly onto output frames.
* **Video Generation**: Processes video frame-by-frame and writes fully annotated output videos with matching frame rates and dimensions.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.8+
* **Object Detection**: PyTorch, YOLOv5 (`ultralytics/yolov5`)
* **OCR Engine**: EasyOCR
* **Computer Vision**: OpenCV (`cv2`)
* **Data Processing**: NumPy, Pandas

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python installed. You can install all required dependencies using `pip`:

```bash
pip install torch torchvision numpy opencv-python easyocr pandas
