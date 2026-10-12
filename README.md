# Privacy-Preserving Licence Plate Redaction Tool (PPLPRT)

A working Term 3 implementation. Upload a JPG/PNG, the app detects visible
licence plates and applies Gaussian blur, then shows a side-by-side
comparison with a download button.

## How this maps to your project documents

| Document element | Implementation |
|---|---|
| Functional Decomposition (Fig. 4) | `redaction.py` functions map 1:1 to each leaf node (Validate Image File, Detect Licence Plates, Filter Low-Confidence Detections, Expand Bounding Box, Apply Gaussian Blur, Generate Redacted Image) |
| UML Class Diagram (Fig. 2) | `MediaFile` class in `redaction.py`; `RedactionEngine` responsibilities implemented as the `redact_image()` pipeline; `User`/`ComparisonReport` responsibilities implemented in `app.py` + `result.html` |
| HLD (Fig. 5) | `app.py` (UI subsystem) → `redaction.py` (Detection + Anonymisation subsystems) → `result.html` (Output subsystem) |
| LLD (Fig. 6) | `redact_image()` follows the exact flow: validate → detect → filter → map boxes → expand box → blur (loop) → generate output → display comparison |
| Functional Requirements (5.2) | All 10 requirements implemented (see checklist below) |

## Requirements checklist (Section 5.2)

- [x] Accept image file uploads
- [x] Validate the uploaded files (extension, size, decodability)
- [x] Provide feedback for invalid/unsupported files
- [x] Detect all visible licence plates
- [x] Create bounding boxes
- [x] Expand the bounding boxes (padding, clipped to image edge)
- [x] Blur licence plates (Gaussian, kernel scaled to box size)
- [x] Generate a processed version of the media
- [x] Display original and processed media side-by-side
- [x] Allow users to download the processed output

## Important note on the detector

Your report specifies a YOLOv8 model trained on CCPD (Chinese plates).
Training that model requires the CCPD dataset (~250k images) and a GPU,
neither of which is available in this sandbox. To ship a **working**
prototype now, this build uses OpenCV's built-in Haar cascade plate
detector (`haarcascade_russian_plate_number.xml`, bundled with
`opencv-python`, no internet/download required, no license/legal
concerns). It performs the same role in the pipeline (bounding boxes +
confidence weight) and is a drop-in swap.

**To upgrade to your actual YOLOv8/CCPD detector later**, only
`redaction.py`'s `detect_plates()` function needs to change — everything
else (Flask app, validation, blur, UI, download) stays the same. That
swap is the natural next Term 3 task once you have trained weights
(`best.pt`) — see "Next steps" below.

## Running it locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## Project structure

```
pplprt/
├── app.py            # Flask routes (UI + Output subsystems)
├── redaction.py       # Detection + Anonymisation engine (RedactionEngine)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html     # Upload page
│   └── result.html    # Side-by-side comparison + download
├── static/
│   └── style.css
├── uploads/            # Temporary storage for uploaded originals
└── outputs/            # Generated comparison + redacted images
```

## Next steps (see chat for full guidance on when to use Claude Code)

1. Swap in your trained YOLOv8 CCPD model in `detect_plates()`.
2. Add automated test suite (Testing Document deliverable) with a labeled
   test image set covering edge cases (angled plates, low light, no plate).
3. Add basic session cleanup (delete uploads/outputs after N minutes).
4. Optional: containerise with Docker for the Operating Requirements section.
