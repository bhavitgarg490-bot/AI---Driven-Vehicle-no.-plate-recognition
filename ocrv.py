import torch
import numpy as np
import cv2
import easyocr
import pandas as pd


class CarDetection:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cpu'
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR
        print("Using device:", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # use yolov5s for smaller model
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes_and_read_text(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        detected_text = []
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:  # confidence threshold
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                #cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            #(255, 255, 255), 2)

                # Perform OCR on the detected region
                roi = frame[y1:y2, x1:x2]
                ocr_results = self.reader.readtext(roi)
                for (bbox, text, prob) in ocr_results:
                    (top_left, top_right, bottom_right, bottom_left) = bbox

                    # Convert top_left to integer coordinates
                    top_left = tuple(map(int, top_left))

                    cv2.putText(frame, text, (top_left[0] + x1, top_left[1] + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)
                    print(f"Detected text: {text} with confidence: {prob}")
                    detected_text.append(text)

        return frame, detected_text

    def class_to_label(self, class_idx):
        return self.classes[int(class_idx)]

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        assert image is not None, "Image not found!"
        frame = cv2.resize(image, (416, 416))
        results = self.score_frame(frame)
        frame, detected_text = self.plot_boxes_and_read_text(results, frame)

        # Save the output image
        output_path = 'output.jpg'
        cv2.imwrite(output_path, frame)
        print(f"Output image saved to {output_path}")

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error opening video stream or file"
        print(f"Processing video: {video_path}")

        # Get video width, height, and fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_count}")
            frame_count += 1

            frame_resized = cv2.resize(frame, (416, 416))
            results = self.score_frame(frame_resized)
            frame_resized, detected_text = self.plot_boxes_and_read_text(results, frame_resized)
            frame_resized = cv2.resize(frame_resized, (width, height))  # Resize back to original size

            out.write(frame_resized)

        cap.release()
        out.release()
        print(f"Processed video saved to {output_path}")


# Example usage
if __name__ == "__main__":
    model_path = 'best.pt'  # Adjust this path to your YOLOv5 model file
    video_path = 'thcar.mp4'  # Adjust this path to your input video file
    output_path = 'output_video.mp4'  # Adjust this path to your output video file

    detector = CarDetection(model_name=model_path)
    detector.process_video(video_path, output_path)
