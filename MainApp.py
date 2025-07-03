import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMessageBox
)
from ultralytics import YOLO

class GarbageDetectionMainApp(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.setWindowTitle("Garbage Detection UI")
        self.resize(800, 600)
        
        # Initialize model
        self.model_path = model_path
        self.model = YOLO(model_path, task='detect')
        self.labels = self.model.names
        
        # Camera state
        self.camera_active = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        
        # Bounding box colors (Tableau 10 scheme)
        self.bbox_colors = [
            (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
        ]
        
        # FPS calculation
        self.avg_frame_rate = 0
        self.frame_rate_buffer = []
        self.fps_avg_len = 200
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #aaa;")
        self.image_label.setMinimumSize(640, 480)
        
        # Buttons
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.select_image)
        
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        
        # Stats display
        self.stats_label = QLabel("Objects detected: 0 | FPS: 0.00")
        
        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.camera_btn)
        btn_layout.addWidget(self.save_btn)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.image_label, 1)
        main_layout.addWidget(self.stats_label)
    
    def select_image(self):
        """Open a file dialog and display the chosen image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if not file_path:
            return
            
        self.process_image(file_path)
    
    def process_image(self, file_path):
        """Process and display an image with object detection"""
        # Read image
        frame = cv2.imread(file_path)
        if frame is None:
            QMessageBox.warning(self, "Error", "Unable to load image")
            return
            
        # Process and display
        processed_frame, object_count = self.detect_objects(frame)
        self.display_image(processed_frame)
        self.stats_label.setText(f"Objects detected: {object_count} | Mode: Image")
        self.save_btn.setEnabled(True)
    
    def toggle_camera(self):
        """Start/stop camera feed"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Initialize and start camera"""
        try:
            self.cap = cv2.VideoCapture(0)  # Default camera
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Unable to open camera")
                return
                
            self.camera_active = True
            self.camera_btn.setText("Stop Camera")
            self.timer.start(30)  # ~30 FPS
            self.stats_label.setText("Objects detected: 0 | Mode: Camera")
            self.frame_rate_buffer = []
            
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera:\n{str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_active = False
        self.camera_btn.setText("Start Camera")
    
    def process_frame(self):
        """Process a frame from the camera"""
        if not self.cap:
            return
            
        try:
            t_start = time.perf_counter()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                self.stop_camera()
                return
            
            # Process and display
            processed_frame, object_count = self.detect_objects(frame)
            self.display_image(processed_frame)
            
            # Update stats
            self.stats_label.setText(f"Objects detected: {object_count} | FPS: {self.avg_frame_rate:.2f} | Mode: Camera")
            self.save_btn.setEnabled(True)
            
            # Calculate FPS
            t_stop = time.perf_counter()
            frame_rate_calc = float(1/(t_stop - t_start))

            # Update FPS buffer
            if len(self.frame_rate_buffer) >= self.fps_avg_len:
                self.frame_rate_buffer.pop(0)
            self.frame_rate_buffer.append(frame_rate_calc)

            # Calculate average FPS
            self.avg_frame_rate = np.mean(self.frame_rate_buffer)
            
        except Exception as e:
            print(f"Camera error: {e}")
            self.stop_camera()
    
    def detect_objects(self, frame):
        """Run object detection on a frame matching deploy.py logic"""
        # Run inference on frame
        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        # Initialize variable for object counting
        object_count = 0

        # Go through each detection
        for i in range(len(detections)):
            # Get bounding box coordinates
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = self.labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > 0.5:
                color = self.bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, 
                             (xmin, label_ymin - labelSize[1] - 10),
                             (xmin + labelSize[0], label_ymin + baseLine - 10),
                             color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Count objects
                object_count += 1
                
        return frame, object_count
    
    def display_image(self, frame):
        """Display OpenCV image in QLabel"""
        # Convert color format (BGR to RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage and display
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.width(), 
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def save_result(self):
        """Save the current displayed image"""
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            str(Path.home() / "detection_result.png"),
            "Images (*.png);;All files (*)",
        )
        if file_path:
            pixmap.save(file_path, "PNG")
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        self.stop_camera()
        event.accept()


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file', required=True)
    args = parser.parse_args()
    
    # Check model file
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        return
        
    # Create and run application
    app = QApplication(sys.argv)
    window = GarbageDetectionMainApp(args.model)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()