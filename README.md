# Green Coverage ROI App

This project measures how much green vegetation is inside a selected region of
an image. It is a practical image-analysis tool, not an ML training project.

## Why this was made

Field images contain a PVC frame or another area that needs to be measured.
The app lets a user draw that area directly, then reports:

- green area as a percentage;
- non-green area as a percentage.

Bright reflections on leaves are handled conservatively when they are small and
surrounded by green pixels.

## Run the app

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app reads images from the [`images`](images) folder.
To use other images, copy JPG, JPEG, or PNG files into that folder.

## Use the app

1. Select an image.
2. Draw one polygon around the area to measure.
3. Click **Analyze ROI**.
4. Read the green and non-green percentages.
5. Download the Excel results when needed.

Results are saved in [`results/roi_results.xlsx`](results/roi_results.xlsx).
If an image is analyzed again, its previous row is replaced.

The colored review overlay is created temporarily during analysis and deleted
automatically. No overlay files are kept.

## Project files

- [`app.py`](app.py): Streamlit user interface.
- [`images`](images): input image folder.
- [`results`](results): final Excel output.
- [`run_green_roi_analysis.py`](run_green_roi_analysis.py):
  reusable green/non-green analysis logic.
