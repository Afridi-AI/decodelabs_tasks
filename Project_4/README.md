# Project 4 — Image or Text Recognition (Basic)
**DecodeLabs AI Training Kit — Path 1: OCR**

## What this is
A working recognition pipeline that reads text out of an image using a
pre-trained OCR engine (Google's Tesseract, via `pytesseract`), exactly as
described in the Project 4 brief.

## Files
- `ocr_pipeline.py` — the full pipeline
- `sample_input.png` — sample scanned-style document (rotated + noisy, to
  simulate a real-world scan)
- `output_annotated.png` — visual confirmation: bounding boxes + confidence
  labels drawn over the recognized text
- `run_log.txt` — console output from the run (recognized text, per-word
  confidence, accuracy benchmark)

## How to run
```bash
pip install opencv-python pytesseract pillow numpy
python3 ocr_pipeline.py
```

## How it satisfies the Gatekeeper Rule (4 validations)
1. **Library Integration** — clean, error-free use of `pytesseract` for
   recognition and `cv2` (OpenCV) for image handling.
2. **Pre-Processing Integrity** — full chain implemented: grayscale
   conversion → Gaussian blur → deskew → Otsu adaptive thresholding, so the
   model reads a clean binary image instead of raw noisy pixels.
3. **Accuracy Benchmarking** — every detection carries a confidence score;
   an **80% confidence gate** (`apply_confidence_gate`) drops anything below
   the minimum standard before it reaches the output. On the sample input,
   average accepted confidence was **95%**.
4. **Visual Confirmation** — `output_annotated.png` shows green boxes +
   text/confidence labels for every accepted detection (red boxes mark
   anything that was filtered out by the 80% gate).

## Design notes
- `--psm 6` is used since the sample input is a single uniform block of
  text (a document page), per the PSM guidance in the brief.
- Confidence-gate logic mirrors the brief's own pseudocode:
  ```python
  if confidence >= 0.80:
      draw_box_and_label()
  else:
      drop_detection()
  ```
- Swap in your own image by replacing `sample_input.png`, or change
  `INPUT_PATH` in `ocr_pipeline.py`.
