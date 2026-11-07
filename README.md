# Green Coverage ROI Desktop App

## For people who do not code: open the app

These steps are for Windows.

1. Install Python from [python.org](https://www.python.org/downloads/). During
   installation, tick **Add Python to PATH**.
2. Download this project and open its folder.
3. Double-click `start_app.bat` (or open PowerShell in this folder and run
   `python desktop_app.py`).
4. Choose a dataset and an image folder on the left.
5. Choose an image, then click at least three points around the vegetation
   region. The yellow line shows your polygon.
6. Click **Analyze ROI**. The result is saved automatically as
   `roi_results.xlsx` in the same folder as that image.
7. Repeat for more images. The app checks for newly added folders and images
   automatically; **Refresh folders and images** is also available.

If Python is not installed yet, run these commands once in PowerShell from the
project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python desktop_app.py
```

## Organising your data

Put field data under one major folder named `Data`:

```text
Data/
├── roadside transect/
│   ├── T1/
│   └── T2/
└── premonsoon/
    ├── C1 COMPLETE/
    │   ├── C1S1/
    │   └── C1S2/
    └── T1 COMPLETE/
        └── T1S1/
```

The app automatically lists every dataset folder and every nested folder that
contains JPG, JPEG, or PNG images. If another dataset folder is added under
`Data`, it appears in the app without a code change. Existing installations
using `ROADSIDE TRANSECT` or `VEGETATION PHOTOS_Illambazar premonsoon 2026`
are also supported as a transition.

The tracked `images/` folder contains small example images for trying the
polygon tool. Original field datasets are local and are intentionally not
stored in Git.

## What the analysis does

The app estimates green and non-green coverage inside a user-drawn polygon,
with a conservative correction for small bright leaf reflections. Results are
written to Excel; re-analysing an image replaces its previous row in that
folder's workbook.

## Citation and authorship

If you use or adapt this software, please cite it as:

> Armu-Aru. *Green Coverage ROI Desktop App*. 2026. Git repository:
> https://github.com/Armu-Aru/habitat-sampling

**Authorship and contributions**

- **Armu-Aru** — project author, field-data workflow, application
  requirements, and repository maintainer.
- **GitHub Copilot** — software implementation assistance and documentation
  assistance, recorded in the Git commit trailers.
- **OpenCV, NumPy, pandas, openpyxl, and PySide6** — third-party software
  dependencies used by this project; please retain their licenses and cite
  them when required by your institution.

Please add the names of field-data collectors, scientific supervisors, and
other contributors to this section before formal publication if they are not
already represented in the repository history.

## Developer notes

This is a Windows desktop app, not an ML training project. Run it with
`python desktop_app.py` after installing the dependencies in
`requirements.txt`.
