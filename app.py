"""Streamlit app for drawing an ROI and measuring green coverage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from run_green_roi_analysis import (
    _classify_green,
    _correct_leaf_glare,
    _overlay,
)


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
RESULTS_DIR = ROOT / "results"
OVERLAY_DIR = ROOT / "Overlay"
EXCEL_PATH = RESULTS_DIR / "roi_results.xlsx"
DISPLAY_WIDTH = 900


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _canvas_image(image: np.ndarray) -> tuple[Image.Image, float]:
    height, width = image.shape[:2]
    scale = min(1.0, DISPLAY_WIDTH / width)
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), scale


def _polygon_from_canvas(canvas: object, scale: float) -> np.ndarray:
    if canvas is None or not hasattr(canvas, "json_data"):
        raise ValueError("Draw a polygon around the area to measure.")
    objects = canvas.json_data.get("objects", [])
    polygons = [
        item
        for item in objects
        if item.get("type") == "polygon" and len(item.get("points", [])) >= 3
    ]
    if not polygons:
        raise ValueError("Draw one polygon around the area to measure.")
    points = np.asarray(
        [[point["x"], point["y"]] for point in polygons[-1]["points"]],
        dtype=np.float32,
    )
    points /= scale
    return np.rint(points).astype(np.int32)


def _roi_mask(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.reshape((-1, 1, 2))], 255)
    return mask


def _save_excel(row: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if EXCEL_PATH.is_file():
        results = pd.read_excel(EXCEL_PATH)
        results = results[results["image"] != row["image"]]
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    else:
        results = pd.DataFrame([row])
    results.to_excel(EXCEL_PATH, index=False)


def main() -> None:
    st.set_page_config(page_title="Green Coverage ROI", layout="wide")
    st.title("Green versus non-green ROI measurement")
    st.write(
        "Select an image, draw one polygon around the area to measure, and "
        "save the percentage result."
    )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ) if IMAGE_DIR.is_dir() else []
    if not image_paths:
        st.error(f"No images found in {IMAGE_DIR}")
        return

    selected_name = st.selectbox("Image", [path.name for path in image_paths])
    image_path = IMAGE_DIR / selected_name
    image = _read_image(image_path)
    display_image, scale = _canvas_image(image)

    canvas = st_canvas(
        fill_color="rgba(0, 0, 255, 0.15)",
        stroke_width=3,
        stroke_color="#0000FF",
        background_image=display_image,
        update_streamlit=True,
        height=display_image.height,
        width=display_image.width,
        drawing_mode="polygon",
        key=f"roi-{selected_name}",
    )
    st.caption("Draw one polygon around the ROI, then click Analyze.")

    if st.button("Analyze ROI", type="primary"):
        try:
            polygon = _polygon_from_canvas(canvas, scale)
            roi_mask = _roi_mask(polygon, image.shape[:2])
            raw_green, threshold = _classify_green(image, roi_mask)
            green, corrected_glare, corrected_pixels = _correct_leaf_glare(
                image, roi_mask, raw_green
            )
            roi_pixels = int(np.count_nonzero(roi_mask))
            green_pixels = int(np.count_nonzero(green))
            overlay = _overlay(image, roi_mask, green, corrected_glare)
            OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
            overlay_path = OVERLAY_DIR / image_path.name
            if not cv2.imwrite(str(overlay_path), overlay):
                raise OSError(f"Unable to write temporary overlay: {overlay_path}")
            success, encoded = cv2.imencode(".jpg", overlay)
            if not success:
                raise OSError("Unable to encode the temporary overlay.")
            row = {
                "image": image_path.name,
                "roi_pixels": roi_pixels,
                "green_percent": round(green_pixels / roi_pixels * 100, 2),
                "non_green_percent": round(
                    (roi_pixels - green_pixels) / roi_pixels * 100, 2
                ),
                "glare_corrected_pixels": corrected_pixels,
                "exg_threshold": round(threshold, 2),
            }
            _save_excel(row)
            st.session_state["last_result"] = row
            st.session_state["last_overlay"] = encoded.tobytes()
        except (OSError, ValueError, KeyError) as error:
            st.error(str(error))
        finally:
            if "overlay_path" in locals() and overlay_path.is_file():
                overlay_path.unlink()

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.success(
            f"{result['image']}: {result['green_percent']}% green, "
            f"{result['non_green_percent']}% non-green"
        )
        st.image(st.session_state["last_overlay"], caption="Temporary review overlay")
        if EXCEL_PATH.is_file():
            st.download_button(
                "Download Excel results",
                data=EXCEL_PATH.read_bytes(),
                file_name="roi_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
