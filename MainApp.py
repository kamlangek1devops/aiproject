import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

class ImageUploader(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Uploader")
        self.resize(600, 450)

        # --- Widgets --------------------------------------------------------
        self.upload_btn = QPushButton("Upload image…")
        self.upload_btn.clicked.connect(self.select_image)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px dashed #aaa;")
        self.image_label.setMinimumSize(300, 200)
        self.image_label.setScaledContents(True)   # auto-fit image

        # --- Layout ---------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.image_label, 1)      # stretch factor = 1

    # ----------------------------------------------------------------------
    def select_image(self):
        """Open a file dialog and display the chosen image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All files (*)",
        )
        if not file_path:          # User pressed Cancel
            return

        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image_label.setText("❌  Unable to load image.")
            return

        self.image_label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageUploader()
    window.show()
    sys.exit(app.exec())
