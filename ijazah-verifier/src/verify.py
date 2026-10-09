"""Prototype verifikasi ijazah: OCR nomor ijazah + deteksi tanda tangan.

Pemakaian:
    python src/verify.py data/samples/01_HighQuality_Enhanced.jpg
    python src/verify.py ijazah_001.jpg --debug results/debug
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
import enhance  # noqa: E402
from ocr import read_number  # noqa: E402
from signature import detect_signature  # noqa: E402

# ROI dalam pecahan (x0, y0, x1, y1) dari citra yang sudah tegak -> tahan terhadap perbedaan resolusi
ROI_NOMOR = (0.03, 0.88, 0.32, 0.965)
ROI_REKTOR = (0.12, 0.715, 0.42, 0.815)
ROI_DEKAN = (0.62, 0.685, 0.90, 0.815)

NOMOR_ENHANCE = "median"  # dipilih berdasarkan eksperimen CER (lihat README)


def crop(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def load_upright(path, rotate="auto"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if rotate == "auto":
        # ijazah berformat landscape; bila tersimpan portrait, putar 90 derajat searah jarum jam
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "90":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "180":
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate == "270":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def verify(path, rotate="auto", nomor_method=NOMOR_ENHANCE, debug_dir=None):
    img = load_upright(path, rotate)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                 # 1. Grayscale
    gray = enhance.METHODS[enhance.GLOBAL_ENHANCE](gray)         # 2. Image enhancement (global)

    # 3a. Cabang nomor ijazah: crop -> enhancement khusus -> OCR
    roi_n = crop(gray, ROI_NOMOR)
    roi_n_enh = enhance.METHODS[nomor_method](roi_n)
    nomor, raw = read_number(roi_n_enh)

    # 3b. Cabang tanda tangan: crop -> thresholding -> morfologi -> deteksi
    sig = {}
    masks = {}
    for name, roi in (("rektor", ROI_REKTOR), ("dekan", ROI_DEKAN)):
        present, info, mask = detect_signature(crop(gray, roi))
        sig[name] = (present, info)
        masks[name] = mask

    all_present = all(p for p, _ in sig.values())
    result = {
        "nomor_ijazah": nomor,
        "ocr_raw": raw,
        "tanda_tangan": "PRESENT" if all_present else "ABSENT",
        "detail": {k: {"present": p, **i} for k, (p, i) in sig.items()},
    }

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"{base}_nomor.png"), roi_n_enh)
        for k, m in masks.items():
            cv2.imwrite(os.path.join(debug_dir, f"{base}_{k}_mask.png"), m)
    return result


def main():
    ap = argparse.ArgumentParser(description="Verifikasi ijazah (OCR nomor + deteksi tanda tangan)")
    ap.add_argument("image")
    ap.add_argument("--rotate", default="auto", choices=["auto", "0", "90", "180", "270"])
    ap.add_argument("--method", default=NOMOR_ENHANCE, choices=list(enhance.METHODS))
    ap.add_argument("--debug", metavar="DIR", help="simpan crop/mask hasil antara")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    r = verify(a.image, a.rotate, a.method, a.debug)
    print(f"Nomor Ijazah : {r['nomor_ijazah'] or '(tidak terbaca)'}")
    print(f"Tanda Tangan : {r['tanda_tangan']}")
    if a.verbose:
        print(f"  OCR mentah : {r['ocr_raw']!r}")
        for k, v in r["detail"].items():
            print(f"  {k:7s}: present={v['present']} ink={v['ink_ratio']} span={v['max_span_ratio']}")


if __name__ == "__main__":
    main()
