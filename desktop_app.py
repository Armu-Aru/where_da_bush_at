"""Native desktop application for green coverage ROI measurement."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
DATA_ROOT = ROOT / "Data"
LEGACY_DATA_ROOTS = {
    "roadside transect": ROOT / "ROADSIDE TRANSECT",
    "premonsoon": ROOT / "VEGETATION PHOTOS_Illambazar premonsoon 2026",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageCanvas(QLabel):
    """Display an image and create or edit one ROI polygon."""

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
        self.mode = "idle"
        self.finalized = False
        self.hover_point: QPointF | None = None
        self.drag_index: int | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image: np.ndarray, clear_points: bool = True) -> None:
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
        if clear_points:
            self.points.clear()
            self.finalized = False
            self.mode = "idle"
            self.hover_point = None
        self._update_transform()
        self.update()

    def start_drawing(self) -> None:
        self.points.clear()
        self.finalized = False
        self.mode = "draw"
        self.hover_point = None
        self.setFocus()
        self.update()

    def start_editing(self) -> None:
        if self.finalized:
            self.mode = "edit"
            self.setFocus()
            self.update()

    def finish_polygon(self) -> bool:
        if len(self.points) < 3:
            return False
        self.finalized = True
        self.mode = "idle"
        self.hover_point = None
        self.drag_index = None
        self.update()
        return True

    def cancel_drawing(self) -> None:
        if not self.finalized:
            self.points.clear()
            self.mode = "idle"
            self.hover_point = None
            self.update()

    def clear_polygon(self) -> None:
        self.points.clear()
        self.finalized = False
        self.mode = "idle"
        self.hover_point = None
        self.drag_index = None
        self.update()

    def _image_point(self, position: QPointF) -> QPointF | None:
        if self._pixmap.isNull() or self.scale <= 0:
            return None
        x = (position.x() - self.offset.x()) / self.scale
        y = (position.y() - self.offset.y()) / self.scale
        if 0 <= x <= self._image_size[0] and 0 <= y <= self._image_size[1]:
            return QPointF(x, y)
        return None

    def _display_point(self, point: QPointF) -> QPointF:
        return QPointF(
            point.x() * self.scale + self.offset.x(),
            point.y() * self.scale + self.offset.y(),
        )

    def _nearest_vertex(self, position: QPointF) -> int | None:
        display_points = [self._display_point(point) for point in self.points]
        for index, point in enumerate(display_points):
            if (point - position).manhattanLength() <= 14:
                return index
        return None

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
        if event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position()
        image_point = self._image_point(position)
        if image_point is None:
            return
        if self.mode == "draw":
            if (
                len(self.points) >= 3
                and (self._display_point(self.points[0]) - position).manhattanLength() <= 14
            ):
                self.finish_polygon()
            else:
                self.points.append(image_point)
                self.update()
        elif self.mode == "edit":
            self.drag_index = self._nearest_vertex(position)

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        image_point = self._image_point(position)
        self.hover_point = image_point if self.mode == "draw" else None
        if self.mode == "edit" and self.drag_index is not None and image_point is not None:
            self.points[self.drag_index] = image_point
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_index = None

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            if self.mode == "draw":
                self.finish_polygon()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_drawing()
        else:
            super().keyPressEvent(event)

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
            pen = QPen(
                Qt.GlobalColor.yellow if not self.finalized else Qt.GlobalColor.green,
                3,
            )
            painter.setPen(pen)
            display_points = [self._display_point(point) for point in self.points]
            for first, second in zip(display_points, display_points[1:]):
                painter.drawLine(first, second)
            if self.finalized and len(display_points) > 2:
                painter.drawLine(display_points[-1], display_points[0])
                painter.setBrush(QBrush(QColor(0, 220, 0, 45)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygonF(display_points))
            painter.setBrush(Qt.GlobalColor.yellow)
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            for point in display_points:
                painter.drawEllipse(point, 5, 5)
            if self.mode == "draw" and self.hover_point is not None:
                painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
                painter.drawLine(display_points[-1], self._display_point(self.hover_point))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Green Coverage ROI")
        self.resize(1200, 800)
        self.dataset_paths: dict[str, Path] = {}
        self.folder_paths: dict[str, Path] = {}
        self.image_paths: list[Path] = []
        self.current_image: np.ndarray | None = None
        self.current_path: Path | None = None
        self.results_path: Path | None = None

        self.dataset_selector = QComboBox()
        self.dataset_selector.currentIndexChanged.connect(self._select_dataset)
        self.folder_selector = QComboBox()
        self.folder_selector.currentIndexChanged.connect(self._select_folder)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._select_image)
        self.canvas = ImageCanvas()

        self.refresh_button = QPushButton("Refresh folders and images")
        self.refresh_button.clicked.connect(self._refresh_data)
        self.draw_button = QPushButton("Create polygon")
        self.draw_button.clicked.connect(self._start_polygon)
        self.edit_button = QPushButton("Edit polygon")
        self.edit_button.clicked.connect(self._edit_polygon)
        self.finish_button = QPushButton("Finish polygon")
        self.finish_button.clicked.connect(self._finish_polygon)
        self.cancel_button = QPushButton("Cancel drawing")
        self.cancel_button.clicked.connect(self._cancel_polygon)
        self.analyze_button = QPushButton("Analyze ROI")
        self.analyze_button.clicked.connect(self._analyze)
        self.clear_button = QPushButton("Clear polygon")
        self.clear_button.clicked.connect(self._clear_polygon)
        self.export_button = QPushButton("Export Excel results")
        self.export_button.clicked.connect(self._export_results)
        self.status = QLabel("Select an image and click points around the ROI.")
        self.status.setWordWrap(True)

        controls = QVBoxLayout()
        controls.addWidget(QLabel("Dataset"))
        controls.addWidget(self.dataset_selector)
        controls.addWidget(QLabel("Image folder"))
        controls.addWidget(self.folder_selector)
        controls.addWidget(self.refresh_button)
        controls.addWidget(QLabel("Images"))
        controls.addWidget(self.image_list)
        controls.addWidget(self.draw_button)
        controls.addWidget(self.edit_button)
        controls.addWidget(self.finish_button)
        controls.addWidget(self.cancel_button)
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
        self._refresh_data()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_data)
        self._refresh_timer.start(2000)

    def _available_datasets(self) -> dict[str, Path]:
        datasets: dict[str, Path] = {}
        if DATA_ROOT.is_dir():
            datasets.update({
                path.name: path
                for path in sorted(DATA_ROOT.iterdir())
                if path.is_dir()
            })
        if not datasets:
            datasets.update({
                name: path
                for name, path in LEGACY_DATA_ROOTS.items()
                if path.is_dir()
            })
        return datasets

    def _refresh_data(self) -> None:
        available_datasets = self._available_datasets()
        selected = self.dataset_selector.currentText()
        selected_folder = self.folder_selector.currentText()
        dataset_changed = available_datasets != self.dataset_paths
        self.dataset_paths = available_datasets
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.addItems(list(self.dataset_paths))
        self.dataset_selector.blockSignals(False)
        if not self.dataset_paths:
            self.folder_selector.clear()
            self.image_list.clear()
            self.status.setText(
                f"No datasets found. Add folders inside {DATA_ROOT} and refresh."
            )
            return
        index = self.dataset_selector.findText(selected)
        self.dataset_selector.setCurrentIndex(max(0, index))
        current_dataset = self.dataset_paths.get(self.dataset_selector.currentText())
        folders_changed = (
            current_dataset is not None
            and self._folder_paths_for(current_dataset) != self.folder_paths
        )
        if dataset_changed or folders_changed or selected != self.dataset_selector.currentText():
            self._select_dataset(self.dataset_selector.currentIndex())
        elif selected_folder in self.folder_paths:
            self.folder_selector.setCurrentText(selected_folder)
            folder_path = self.folder_paths[selected_folder]
            folder_images = sorted(
                path.name
                for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if folder_images != [path.name for path in self.image_paths]:
                self._select_folder(self.folder_selector.currentIndex())
        else:
            self._select_dataset(self.dataset_selector.currentIndex())

    def _folder_paths_for(self, dataset_path: Path) -> dict[str, Path]:
        folder_paths: dict[str, Path] = {}
        if any(
            child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
            for child in dataset_path.iterdir()
        ):
            folder_paths["(this folder)"] = dataset_path
        folder_paths.update({
            str(path.relative_to(dataset_path)): path
            for path in sorted(dataset_path.rglob("*"))
            if path.is_dir()
            and any(
                child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
                for child in path.iterdir()
            )
        })
        return folder_paths

    def _select_dataset(self, index: int) -> None:
        if index < 0:
            return
        dataset_path = self.dataset_paths.get(self.dataset_selector.currentText())
        if dataset_path is None:
            return
        selected_folder = self.folder_selector.currentText()
        self.folder_paths = self._folder_paths_for(dataset_path)
        self.folder_selector.blockSignals(True)
        self.folder_selector.clear()
        self.folder_selector.addItems(list(self.folder_paths))
        self.folder_selector.blockSignals(False)
        if self.folder_paths:
            folder_index = self.folder_selector.findText(selected_folder)
            self.folder_selector.setCurrentIndex(max(0, folder_index))
            self._select_folder(self.folder_selector.currentIndex())
        else:
            self.image_list.clear()
            self.status.setText(f"No images found inside {dataset_path}")

    def _select_folder(self, index: int) -> None:
        if index < 0:
            return
        folder_path = self.folder_paths.get(self.folder_selector.currentText())
        if folder_path is None:
            return
        selected_image = self.image_list.currentItem().text() if self.image_list.currentItem() else ""
        self.image_paths = sorted(
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self.image_list.addItems([path.name for path in self.image_paths])
        self.image_list.blockSignals(False)
        if self.image_paths:
            image_index = next(
                (i for i, path in enumerate(self.image_paths) if path.name == selected_image),
                0,
            )
            self.image_list.setCurrentRow(image_index)
        else:
            self.status.setText(f"No images found in {folder_path}")

    def _select_image(self, index: int) -> None:
        if index < 0 or index >= len(self.image_paths):
            return
        self.current_path = self.image_paths[index]
        self.results_path = self.current_path.parent / "roi_results.xlsx"
        self.current_image = cv2.imread(str(self.current_path), cv2.IMREAD_COLOR)
        if self.current_image is None:
            self.status.setText(f"Unable to read {self.current_path}")
            return
        self.canvas.set_image(self.current_image)
        self.status.setText(
            f"{self.current_path.name}: click Create polygon, then click around the ROI."
        )

    def _start_polygon(self) -> None:
        if self.current_image is None:
            self.status.setText("Select an image first.")
            return
        self.canvas.start_drawing()
        self.status.setText(
            "Drawing polygon: click points, click the first point again or press "
            "Enter to finish. Press Escape to cancel."
        )

    def _edit_polygon(self) -> None:
        if not self.canvas.finalized:
            self.status.setText("Finish a polygon before editing it.")
            return
        self.canvas.start_editing()
        self.status.setText("Edit mode: drag a yellow vertex to adjust the polygon.")

    def _finish_polygon(self) -> None:
        if self.canvas.finish_polygon():
            self.status.setText("Polygon finished. Click Analyze ROI to run analysis.")
        else:
            self.status.setText("A polygon needs at least three points.")

    def _cancel_polygon(self) -> None:
        self.canvas.cancel_drawing()
        self.status.setText("Drawing canceled.")

    def _clear_polygon(self) -> None:
        self.canvas.clear_polygon()
        self.status.setText("Polygon cleared. Click Create polygon to draw again.")

    def _analyze(self) -> None:
        if self.current_image is None or self.current_path is None:
            self.status.setText("Select an image first.")
            return
        if not self.canvas.finalized:
            self.status.setText("Finish the polygon before analyzing it.")
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
        self.canvas.set_image(overlay, clear_points=False)

    def _save_result(self, row: dict[str, object]) -> None:
        if self.current_path is None:
            return
        self.results_path = self.current_path.parent / "roi_results.xlsx"
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        if self.results_path.is_file():
            results = pd.read_excel(self.results_path)
            if "image" not in results.columns:
                results = pd.DataFrame()
            else:
                results = results[results["image"] != row["image"]]
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        else:
            results = pd.DataFrame([row])
        results.to_excel(self.results_path, index=False)

    def _export_results(self) -> None:
        if self.results_path is None or not self.results_path.is_file():
            QMessageBox.information(self, "No results", "Analyze an image first.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export results",
            "roi_results.xlsx",
            "Excel workbook (*.xlsx)",
        )
        if destination:
            Path(destination).write_bytes(self.results_path.read_bytes())
            self.status.setText(f"Exported results to {destination}")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
