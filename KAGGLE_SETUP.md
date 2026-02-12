# Kaggle Setup Instructions — Door Defect Detection

## Quick Start (5 minutes)

### Step 1: Upload Your Dataset to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Name it **`door-defect-data`** (this exact name matters)
3. Upload your 3 folders as-is:
   ```
   door-defect-data/
   ├── black/
   │   ├── data.yaml
   │   └── train/
   │       ├── images/
   │       └── labels/
   ├── white/
   │   ├── data.yaml
   │   └── train/
   │       ├── images/
   │       └── labels/
   └── glossy/
       ├── data.yaml
       └── train/
           ├── images/
           └── labels/
   ```
4. Click **Create** and wait for the upload to finish

> [!TIP]
> You can zip each folder first, Kaggle will auto-extract them.

---

### Step 2: Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **File → Add Data** → search for your dataset `door-defect-data` → **Add**
3. **Enable GPU**: Click the **⋮** menu (top-right) → **Accelerator** → **GPU T4 x2** (or P100)

> [!IMPORTANT]
> GPU is required. Training will be extremely slow on CPU.

---

### Step 3: Run the Script

**Option A — Single cell (simplest)**:
1. Delete the default code cell
2. Copy the entire contents of `kaggle.py` into one cell
3. Click **Run**

**Option B — Upload as utility**:
1. Upload `kaggle.py` to Kaggle as a **Utility Script** dataset
2. In your notebook, add that dataset
3. Run:
   ```python
   exec(open("/kaggle/input/your-script-dataset/kaggle.py").read())
   ```

---

### Step 4: Download Results

After training finishes (~1-2 hours depending on GPU):

1. Go to the **Output** tab on the right sidebar
2. Download:
   - `models/door_defect_detector.pth` — deployment model
   - `results/` — visualizations, heatmaps, JSON reports
   - `runs/` — training curves and metrics

---

## What the Script Does

| Step | Action | Duration |
|------|--------|----------|
| 1 | Merges black + white + glossy datasets | ~30 sec |
| 2 | Camera calibration (default) | instant |
| 3 | Trains YOLOv8n-seg for 200 epochs | ~60-90 min |
| 4 | Validates on test set | ~2 min |
| 5 | Saves deployment model + runs inference | ~5 min |

---

## Output Files

```
/kaggle/working/
├── models/
│   └── door_defect_detector.pth        ← deployment model
├── results/
│   ├── image_name_visualization.jpg    ← 4-panel image
│   ├── image_name_heatmap.jpg          ← standalone heatmap
│   └── image_name_report.json          ← defect measurements
├── runs/
│   └── segment/door_defect_detection/
│       ├── weights/best.pt
│       └── results.csv
└── calibration_config.json
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: black not found` | Check dataset name is exactly `door-defect-data`, and folders are `black/`, `white/`, `glossy/` at root level |
| `CUDA out of memory` | Reduce `batch_size` from 16 to 8 in the `main()` function |
| Training is slow | Make sure GPU is enabled (check Accelerator setting) |
| `ModuleNotFoundError` | The script auto-installs dependencies; if it fails, add `!pip install ultralytics albumentations` as the first cell |

> [!NOTE]
> If your dataset name on Kaggle is different from `door-defect-data`, update the `DATASET_NAME` variable at the top of the script.

---

Made with Love by Neal Daftary
