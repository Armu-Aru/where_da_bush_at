# Green Coverage ROI Desktop App

This Windows desktop app measures green and non-green vegetation inside a
region selected by the user. It is a practical image-analysis tool, not an ML
training project.

## What it does

- Opens the images in the `images` folder.
- Lets you draw an ROI polygon directly on an image.
- Estimates green and non-green percentages inside that polygon.
- Applies a conservative correction for small bright leaf reflections.
- Saves results to an Excel workbook.

## Run it

From the project folder in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python desktop_app.py
```

The app opens as a normal desktop window; no browser is required.

## Use it

1. Put JPG, JPEG, or PNG files in the `images` folder.
2. Select an image from the left-hand list.
3. Click at least three points around the ROI.
4. Click **Analyze ROI**.
5. Read the percentages in the status message.
6. Repeat for other images.
7. Click **Export Excel results** when finished.

Results are stored in `results/roi_results.xlsx`. Re-analyzing an image
replaces its previous row. No overlay images are saved.

## GitHub

This repository contains the desktop app, the reusable analysis algorithm,
requirements, README, and sample input images. Large original field datasets
and generated outputs are intentionally excluded.
