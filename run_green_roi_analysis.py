"""Estimate green versus non-green coverage inside Stage 1 ROI polygons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "results" / "roi_stage1_annotation"
GLARE_VALUE_MIN = 220
GLARE_SATURATION_MAX = 70
GLARE_CHANNEL_SPREAD_MAX = 55
GLARE_MAX_AREA_FRACTION = 0.05
GLARE_SURROUNDING_GREEN_MIN = 0.30


def _roi_mask(label_path: Path, shape: tuple[int, int]) -> np.ndarray:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    polygons = [
        shape_data["points"]
        for shape_data in data.get("shapes", [])
        if shape_data.get("shape_type") == "polygon"
        and len(shape_data.get("points", [])) >= 3
    ]
    if not polygons:
        raise ValueError(f"No polygon found in {label_path}")
    points = np.rint(np.asarray(polygons[0], dtype=np.float32)).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)
    return mask


def _classify_green(image: np.ndarray, roi_mask: np.ndarray) -> tuple[np.ndarray, float]:
    channels = image.astype(np.float32)
    exg = 2.0 * channels[:, :, 1] - channels[:, :, 2] - channels[:, :, 0]
    roi_values = exg[roi_mask > 0]
    if roi_values.size == 0:
        raise ValueError("ROI polygon contains no pixels")
    low = float(roi_values.min())
    high = float(roi_values.max())
    if high <= low:
        threshold = low
    else:
        scaled = np.rint((roi_values - low) * 255 / (high - low)).astype(np.uint8)
        threshold_u8, _ = cv2.threshold(
            scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = low + float(threshold_u8) * (high - low) / 255
    green_mask = ((exg > threshold) & (roi_mask > 0)).astype(np.uint8) * 255
    return green_mask, float(threshold)


def _correct_leaf_glare(
    image: np.ndarray, roi_mask: np.ndarray, green_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Recover small bright glare regions enclosed by nearby green pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channel_min = image.min(axis=2)
    channel_spread = image.max(axis=2) - channel_min
    glare = (
        (hsv[:, :, 2] >= GLARE_VALUE_MIN)
        & (hsv[:, :, 1] <= GLARE_SATURATION_MAX)
        & (channel_spread <= GLARE_CHANNEL_SPREAD_MAX)
        & (roi_mask > 0)
        & (green_mask == 0)
    ).astype(np.uint8)
    roi_pixels = max(1, int(np.count_nonzero(roi_mask)))
    max_component_area = max(20, int(roi_pixels * GLARE_MAX_AREA_FRACTION))
    radius = max(3, round(min(image.shape[:2]) * 0.004))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    green_binary = (green_mask > 0).astype(np.uint8)
    green_nearby = cv2.dilate(green_binary, kernel)
    corrected = green_mask.copy()
    corrected_glare = np.zeros_like(green_mask)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        glare, connectivity=8
    )
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        component = labels == component_id
        if area > max_component_area or not np.any(component & (green_nearby > 0)):
            continue
        expanded = cv2.dilate(component.astype(np.uint8), kernel) > 0
        ring = expanded & ~component & (roi_mask > 0)
        surrounding_green = np.count_nonzero(ring & (green_mask > 0))
        ring_pixels = np.count_nonzero(ring)
        if ring_pixels and surrounding_green / ring_pixels >= GLARE_SURROUNDING_GREEN_MIN:
            corrected[component] = 255
            corrected_glare[component] = 255
    return corrected, corrected_glare, int(np.count_nonzero(corrected_glare))


def _overlay(
    image: np.ndarray,
    roi_mask: np.ndarray,
    green_mask: np.ndarray,
    corrected_glare: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    inside = roi_mask > 0
    non_green = inside & (green_mask == 0)
    green = green_mask > 0
    tint = np.zeros_like(image)
    tint[non_green] = (0, 0, 210)
    tint[green] = (0, 180, 0)
    tint[corrected_glare > 0] = (0, 220, 220)
    result[inside] = cv2.addWeighted(image[inside], 0.45, tint[inside], 0.55, 0)
    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.polylines(
        result,
        contours,
        True,
        (255, 0, 0),
        2,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate green and non-green pixels inside Stage 1 ROIs."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    args = parser.parse_args()

    image_dir = args.workspace / "images"
    label_dir = args.workspace / "labels"
    output_dir = args.workspace / "green_analysis"
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for label_path in sorted(label_dir.glob("*.json")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image for {label_path}: {image_path}")
        roi_mask = _roi_mask(label_path, image.shape[:2])
        raw_green_mask, threshold = _classify_green(image, roi_mask)
        green_mask, corrected_glare, corrected_pixels = _correct_leaf_glare(
            image, roi_mask, raw_green_mask
        )
        roi_pixels = int(np.count_nonzero(roi_mask))
        raw_green_pixels = int(np.count_nonzero(raw_green_mask))
        green_pixels = int(np.count_nonzero(green_mask))
        non_green_pixels = roi_pixels - green_pixels
        overlay_path = overlay_dir / image_path.name
        if not cv2.imwrite(
            str(overlay_path),
            _overlay(image, roi_mask, green_mask, corrected_glare),
        ):
            raise OSError(f"Unable to write {overlay_path}")
        rows.append(
            {
                "image": image_path.name,
                "roi_pixels": roi_pixels,
                "raw_green_pixels": raw_green_pixels,
                "glare_corrected_pixels": corrected_pixels,
                "green_pixels": green_pixels,
                "non_green_pixels": non_green_pixels,
                "raw_green_percent": round(raw_green_pixels / roi_pixels * 100, 2),
                "green_percent": round(green_pixels / roi_pixels * 100, 2),
                "non_green_percent": round(non_green_pixels / roi_pixels * 100, 2),
                "exg_threshold": round(threshold, 2),
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Analyzed {len(rows)} Stage 1 ROIs")
    print(f"Summary: {summary_path}")
    print(f"Review overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
