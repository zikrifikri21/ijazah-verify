"""Eksperimen:
 1. CER OCR nomor ijazah per metode enhancement pada 9 citra sampel.
 2. Uji stres: degradasi sintetis bertingkat (seed tetap) agar metode yang seri di (1) dapat dibedakan.
 3. Kontrol negatif deteksi tanda tangan (tanda tangan dihapus secara sintetis).

    python src/evaluate.py --samples data/samples --truth 571012022000056 --out results
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import enhance  # noqa: E402
from ocr import cer, read_number  # noqa: E402
from signature import detect_signature  # noqa: E402
from verify import ROI_DEKAN, ROI_NOMOR, ROI_REKTOR, crop, load_upright  # noqa: E402


# ---------------------------------------------------------------- 1. CER pada sampel
def run_cer(files, truth):
    rows = []
    for f in files:
        gray = cv2.cvtColor(load_upright(f), cv2.COLOR_BGR2GRAY)
        roi = crop(gray, ROI_NOMOR)
        for name, fn in enhance.METHODS.items():
            pred, _ = read_number(fn(roi))
            rows.append({"image": os.path.basename(f), "method": name, "pred": pred,
                         "CER": cer(pred, truth)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. uji stres
def degrade(g, kind, level, rng):
    g = g.astype(np.float32)
    if kind == "noise":
        g = g + rng.normal(0, level, g.shape)
    elif kind == "blur":
        g = cv2.GaussianBlur(g, (0, 0), level)
    elif kind == "lowcontrast":            # level = faktor kontras; kertas dibuat pudar
        g = 255 - (255 - g) * level
        g = g * 0.85
    elif kind == "dark":                   # underexposed
        g = g * level
    elif kind == "lowres":                 # level = faktor skala turun lalu naik lagi
        h, w = g.shape
        g = cv2.resize(cv2.resize(g, None, fx=level, fy=level, interpolation=cv2.INTER_AREA),
                       (w, h), interpolation=cv2.INTER_LINEAR)
    g = np.clip(g, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        g = cv2.imdecode(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(level)])[1], 0)
    return g


STRESS = [("noise", 15), ("noise", 30), ("noise", 50),
          ("blur", 1.5), ("blur", 2.5), ("blur", 3.5),
          ("lowcontrast", 0.35), ("lowcontrast", 0.2),
          ("dark", 0.5), ("dark", 0.3),
          ("lowres", 0.35), ("lowres", 0.2),
          ("jpeg", 15), ("jpeg", 6)]


def run_stress(base_file, truth, seed=0):
    gray = cv2.cvtColor(load_upright(base_file), cv2.COLOR_BGR2GRAY)
    roi = crop(gray, ROI_NOMOR)
    rows = []
    for kind, level in STRESS:
        rng = np.random.default_rng(seed)
        deg = degrade(roi, kind, level, rng)
        for name, fn in enhance.METHODS.items():
            pred, _ = read_number(fn(deg))
            rows.append({"degradasi": f"{kind}={level}", "kind": kind, "method": name,
                         "pred": pred, "CER": cer(pred, truth)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. kontrol negatif tanda tangan
def blank_signature(img_upright, roi):
    out = img_upright.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = roi
    sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
    out[sl] = np.median(out[sl].reshape(-1, 3), axis=0).astype(np.uint8)
    return out


def run_signature_test(files):
    rows = []
    for f in files:
        img = load_upright(f)
        for label, im in (("asli", img),
                          ("rektor_dihapus", blank_signature(img, ROI_REKTOR)),
                          ("dekan_dihapus", blank_signature(img, ROI_DEKAN))):
            g = enhance.METHODS[enhance.GLOBAL_ENHANCE](cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            pr, _, _ = detect_signature(crop(g, ROI_REKTOR))
            pdk, _, _ = detect_signature(crop(g, ROI_DEKAN))
            rows.append({"image": os.path.basename(f), "kondisi": label, "rektor": pr, "dekan": pdk})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.samples, "*")))

    df = run_cer(files, a.truth)
    df.to_csv(os.path.join(a.out, "cer_detail.csv"), index=False)
    df.pivot(index="method", columns="image", values="CER").to_csv(os.path.join(a.out, "cer_matrix.csv"))
    s1 = df.groupby("method").agg(mean_CER=("CER", "mean"), max_CER=("CER", "max"),
                                  n_exact=("CER", lambda s: int((s == 0).sum())))

    base = [f for f in files if "01_" in os.path.basename(f)][0]
    st = run_stress(base, a.truth)
    st.to_csv(os.path.join(a.out, "stress_detail.csv"), index=False)
    st.pivot(index="method", columns="degradasi", values="CER").to_csv(os.path.join(a.out, "stress_matrix.csv"))
    s2 = st.groupby("method").agg(stress_mean_CER=("CER", "mean"),
                                  stress_n_exact=("CER", lambda s: int((s == 0).sum())))

    summ = s1.join(s2).sort_values(["mean_CER", "stress_mean_CER"])
    summ.to_csv(os.path.join(a.out, "cer_summary.csv"))
    print("=== (1) CER pada 9 sampel + (2) uji stres 14 degradasi ===")
    print(summ.round(4).to_string())

    sg = run_signature_test(files)
    sg.to_csv(os.path.join(a.out, "signature_test.csv"), index=False)
    print("\n=== (3) Deteksi tanda tangan: proporsi 'terdeteksi' ===")
    print(sg.groupby("kondisi")[["rektor", "dekan"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
