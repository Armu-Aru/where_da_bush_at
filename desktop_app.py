"""Native desktop application for green coverage ROI measurement."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from run_green_roi_analysis import (
    _classify_green,
    _correct_leaf_glare,
    _overlay,
)


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
RESULTS_PATH = ROOT / "results" / "roi_results.xlsx"


class ImageCanvas(QLabel):
    """Display an image and collect a polygon by mouse clicks."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(720, 520)
        self.setStyleSheet("background: #202020;")
        self._pixmap = QPixmap()
        self._image_size = (0, 0)
        self.points: list[QPointF] = []
        self.scale = 1.0
        self.offset = QPointF()

    def set_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._image_size = (width, height)
        self.points.clear()
        self._update_transform()
        self.update()

    def _update_transform(self) -> None:
        if self._pixmap.isNull():
            return
        self.scale = min(
            self.width() / self._pixmap.width(),
            self.height() / self._pixmap.height(),
        )
        display_width = self._pixmap.width() * self.scale
        display_height = self._pixmap.height() * self.scale
        self.offset = QPointF(
            (self.width() - display_width) / 2,
            (self.height() - display_height) / 2,
        )

    def resizeEvent(self, event) -> None:
        self._update_transform()
        self.update()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._pixmap.isNull():
            return
        position = event.position()
        x = (position.x() - self.offset.x()) / self.scale
        y = (position.y() - self.offset.y()) / self.scale
        if 0 <= x <= self._image_size[0] and 0 <= y <= self._image_size[1]:
            self.points.append(QPointF(x, y))
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self.scale),
            int(self._pixmap.height() * self.scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(self.offset, scaled)
        if self.points:
            pen = QPen(Qt.GlobalColor.blue, 3)
            painter.setPen(pen)
            display_points = [
                QPointF(
                    point.x() * self.scale + self.offset.x(),
                    point.y() * self.scale + self.offset.y(),
                )
                for point in self.points
            ]
            for first, second in zip(display_points, display_points[1:]):
                painter.drawLine(first, second)
            if len(display_points) > 2:
                painter.drawLine(display_points[-1], display_points[0])


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Green Coverage ROI")
        self.resize(1200, 800)
        self.image_paths = sorted(
            path
            for path in IMAGE_DIR.glob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.current_image: np.ndarray | None = None
        self.current_path: Path | None = None

        self.image_list = QListWidget()
        self.image_list.addItems([path.name for path in self.image_paths])
        self.image_list.currentRowChanged.connect(self._select_image)
        self.canvas = ImageCanvas()

        self.analyze_button = QPushButton("Analyze ROI")
        self.analyze_button.clicked.connect(self._analyze)
        self.clear_button = QPushButton("Clear polygon")
        self.clear_button.clicked.connect(self._clear_polygon)
        self.export_button = QPushButton("Export Excel results")
        self.export_button.clicked.connect(self._export_results)
        self.status = QLabel("Select an image and click points around the ROI.")
        self.status.setWordWrap(True)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Images"))
        controls.addWidget(self.image_list)
        controls.addWidget(self.analyze_button)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.status)
        controls.addStretch()

        layout = QHBoxLayout()
        layout.addLayout(controls, 1)
        layout.addWidget(self.canvas, 4)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        if self.image_paths:
            self.image_list.setCurrentRow(0)
        else:
            self.status.setText(f"No images found in {IMAGE_DIR}")

    def _select_image(self, index: int) -> None:
        if index < 0 or index >= len(self.image_paths):
            return
        self.current_path = self.image_paths[index]
        self.current_image = cv2.imread(str(self.current_path), cv2.IMREAD_COLOR)
        if self.current_image is None:
            self.status.setText(f"Unable to read {self.current_path}")
            return
        self.canvas.set_image(self.current_image)
        self.status.setText(
            f"{self.current_path.name}: click at least three points around the ROI."
        )

    def _clear_polygon(self) -> None:
        self.canvas.points.clear()
        self.canvas.update()
        self.status.setText("Polygon cleared. Draw the ROI again.")

    def _analyze(self) -> None:
        if self.current_image is None or self.current_path is None:
            self.status.setText("Select an image first.")
            return
        if len(self.canvas.points) < 3:
            self.status.setText("Add at least three polygon points.")
            return
        points = np.rint(
            [[point.x(), point.y()] for point in self.canvas.points]
        ).astype(np.int32)
        roi_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [points.reshape((-1, 1, 2))], 255)
        raw_green, threshold = _classify_green(self.current_image, roi_mask)
        green, corrected_glare, corrected_pixels = _correct_leaf_glare(
            self.current_image, roi_mask, raw_green
        )
        roi_pixels = int(np.count_nonzero(roi_mask))
        green_pixels = int(np.count_nonzero(green))
        row = {
            "image": self.current_path.name,
            "roi_pixels": roi_pixels,
            "green_percent": round(green_pixels / roi_pixels * 100, 2),
            "non_green_percent": round(
                (roi_pixels - green_pixels) / roi_pixels * 100, 2
            ),
            "glare_corrected_pixels": corrected_pixels,
            "exg_threshold": round(threshold, 2),
        }
        self._save_result(row)
        self.status.setText(
            f"Saved: {row['green_percent']}% green, "
            f"{row['non_green_percent']}% non-green."
        )
        overlay = _overlay(
            self.current_image,
            roi_mask,
            green,
            corrected_glare,
        )
        self.canvas.set_image(overlay)

    def _save_result(self, row: dict[str, object]) -> None:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if RESULTS_PATH.is_file():
            results = pd.read_excel(RESULTS_PATH)
            results = results[results["image"] != row["image"]]
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        else:
            results = pd.DataFrame([row])
        results.to_excel(RESULTS_PATH, index=False)

    def _export_results(self) -> None:
        if not RESULTS_PATH.is_file():
            QMessageBox.information(self, "No results", "Analyze an image first.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export results",
            "roi_results.xlsx",
            "Excel workbook (*.xlsx)",
        )
        if destination:
            Path(destination).write_bytes(RESULTS_PATH.read_bytes())
            self.status.setText(f"Exported results to {destination}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
