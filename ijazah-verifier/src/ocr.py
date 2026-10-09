"""OCR nomor ijazah (Tesseract) + perhitungan CER."""
import re

import cv2
import pytesseract

CFG = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789NomrijazahDSK-.:/ "


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, truth):
    """Character Error Rate = jarak Levenshtein / panjang ground truth."""
    return levenshtein(pred, truth) / max(len(truth), 1)


def extract_number(text):
    """Ambil deret digit terpanjang dari teks hasil OCR."""
    runs = re.findall(r"\d[\d\-\.\s]*\d|\d", text)
    runs = [re.sub(r"\D", "", r) for r in runs]
    runs = [r for r in runs if r]
    return max(runs, key=len) if runs else ""


def read_number(roi_gray, scale_to_height=96):
    """OCR pada ROI; tinggi ROI dinormalkan agar ukuran teks konsisten antar resolusi."""
    h = roi_gray.shape[0]
    f = scale_to_height / h
    img = cv2.resize(roi_gray, None, fx=f, fy=f,
                     interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
    text = pytesseract.image_to_string(img, config=CFG)
    return extract_number(text), text.strip()
