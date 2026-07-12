import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH = "sample_input.png"
OUTPUT_PATH = "output_annotated.png"
CONFIDENCE_THRESHOLD = 80  # The 80% Gate, per project spec
TESSERACT_PSM = 6          # Single uniform block of text (document page)


# ---------------------------------------------------------------------------
# Step 1-3: Pre-processing
# ---------------------------------------------------------------------------
def deskew(gray: np.ndarray) -> np.ndarray:
    """Calculate the rotation angle of the text block and rotate it back
    to a horizontal baseline."""
    # Invert + threshold so text pixels are white on a black background,
    # which is what cv2.minAreaRect expects for angle detection.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grayscale -> Gaussian Blur -> Deskew -> Adaptive Threshold."""
    # Step 1: Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur (smooth micro-imperfections)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Deskewing (snap tilted text to a horizontal baseline)
    straightened = deskew(blurred)

    # Step 4: Adaptive Thresholding (Otsu's method - binary decision)
    _, binary = cv2.threshold(
        straightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, straightened


# ---------------------------------------------------------------------------
# Step 5-6: Recognition + Confidence Gate
# ---------------------------------------------------------------------------
def run_ocr(binary_image: np.ndarray, psm: int = TESSERACT_PSM):
  
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        binary_image, config=config, output_type=Output.DICT
    )

    results = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if text and conf > 0:  # tesseract uses -1 for non-text regions
            results.append({
                "text": text,
                "confidence": conf,
                "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            })
    return results


def apply_confidence_gate(results, threshold=CONFIDENCE_THRESHOLD):
   
    accepted, dropped = [], []
    for r in results:
        if r["confidence"] >= threshold:
            accepted.append(r)
        else:
            dropped.append(r)
    return accepted, dropped


# ---------------------------------------------------------------------------
# Step 7: Visual Confirmation
# ---------------------------------------------------------------------------
def draw_annotations(base_image: np.ndarray, accepted, dropped):
   
    annotated = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

    for r in dropped:
        x, y, w, h = r["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 1)

    for r in accepted:
        x, y, w, h = r["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{r['text']} ({r['confidence']:.0f}%)"
        cv2.putText(
            annotated, label, (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1, cv2.LINE_AA
        )
    return annotated


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    image = cv2.imread(INPUT_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {INPUT_PATH}")

    print(f"[1] Loaded input image: {INPUT_PATH} ({image.shape[1]}x{image.shape[0]})")

    binary, straightened = preprocess(image)
    print("[2] Pre-processing complete: grayscale -> blur -> deskew -> adaptive threshold")

    raw_results = run_ocr(binary)
    print(f"[3] Tesseract OCR complete: {len(raw_results)} text region(s) detected")

    accepted, dropped = apply_confidence_gate(raw_results)
    print(f"[4] Confidence gate ({CONFIDENCE_THRESHOLD}%): "
          f"{len(accepted)} accepted, {len(dropped)} dropped")

    annotated = draw_annotations(binary, accepted, dropped)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"[5] Visual confirmation saved: {OUTPUT_PATH}")

    print("\n--- RECOGNIZED TEXT (>= {}% confidence) ---".format(CONFIDENCE_THRESHOLD))
    full_text = " ".join(r["text"] for r in accepted)
    print(full_text if full_text else "(no text passed the confidence gate)")

    print("\n--- WORD-LEVEL BREAKDOWN ---")
    for r in accepted:
        print(f"  '{r['text']}'  confidence={r['confidence']:.1f}%  box={r['box']}")

    if accepted:
        avg_conf = sum(r["confidence"] for r in accepted) / len(accepted)
        print(f"\n[6] Accuracy benchmark: average confidence = {avg_conf:.1f}% "
              f"(minimum standard: {CONFIDENCE_THRESHOLD}%)")

    return accepted, dropped


if __name__ == "__main__":
    main()
