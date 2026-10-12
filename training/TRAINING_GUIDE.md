# Training your YOLOv8 detector on CCPD

This needs a GPU and internet access to the CCPD dataset, so it has to run
on your own machine or a free cloud notebook -- not something I can run in
this chat sandbox. Google Colab (free tier) is the easiest path if you
don't have a local GPU.

## Step 1 — Get the CCPD dataset

Repo (with dataset download links): https://github.com/detectRecog/CCPD

The full dataset is ~250,000 images (~12GB) hosted on Google Drive / Baidu
Netdisk, linked from that repo's README. For an Honours project on a
timeline, **you don't need the whole thing**. Recommended:

- Download just the `CCPD-Base` subset (or any single weather/condition
  subset like `CCPD-FN` or `CCPD-Tilt` if you want to focus on harder cases).
- You'll subsample further in Step 2 anyway (5,000–15,000 images is plenty
  for a working detector and trains in under an hour on a free Colab GPU).

Unzip it somewhere, e.g. `~/data/ccpd_base/`.

## Step 2 — Convert to YOLO format

Use the script in `training/ccpd_to_yolo.py` (already tested against the
real CCPD filename format):

```bash
pip install pillow
python training/ccpd_to_yolo.py \
    --source ~/data/ccpd_base \
    --dest ~/data/ccpd_yolo \
    --val-split 0.1 \
    --limit 8000
```

This creates:
```
ccpd_yolo/
├── images/train, images/val
└── labels/train, labels/val
```

## Step 3 — Train (on Colab, recommended)

1. Open https://colab.research.google.com, new notebook.
2. Runtime → Change runtime type → GPU (T4 is fine, free tier).
3. Upload your `ccpd_yolo` folder (zip it first, then unzip in Colab —
   uploading 8,000 small images individually is slow) and `data.yaml`
   (copy from `training/data.yaml.template`, edit the `path:` line to
   match where you unzip it in Colab, e.g. `/content/ccpd_yolo`).
4. Run:

```python
!pip install ultralytics -q

from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # nano model -- fastest, good enough for one class
model.train(data="/content/data.yaml", epochs=50, imgsz=640, batch=16)
```

5. When training finishes, your weights are at:
```
runs/detect/train/weights/best.pt
```
Download that file from Colab's file browser (left sidebar).

Training 8,000 images for 50 epochs on a T4 GPU typically takes
20–45 minutes. If Colab disconnects, `runs/detect/train2/weights/best.pt`
etc. will hold the latest run.

## Step 4 — Add it to the project

Drop the downloaded file in as:
```
pplprt/model/best.pt
```

That's it — `redaction.py` checks for this file automatically and switches
from the Haar cascade to your trained model. No other code changes needed.
Restart the Flask app (`python app.py`) and re-test with real photos.

## Step 5 — Sanity check before you rely on it

```python
from ultralytics import YOLO
model = YOLO("model/best.pt")
results = model("path/to/a/test_car_photo.jpg", save=True)
```
This saves an annotated image to `runs/detect/predict/` so you can see the
boxes it drew before trusting it inside the full app. If it's missing
plates or drawing boxes in the wrong place, that usually means: too few
training images, too few epochs, or images that don't resemble your
detection scenario (CCPD is Chinese street-parking scenes — if your test
images are very different, e.g. close-up studio shots, expect weaker
performance and mention this as a limitation in your Testing document).
