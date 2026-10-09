"""Deteksi tanda tangan: thresholding -> morfologi -> analisis komponen terhubung."""
import cv2
import numpy as np


def detect_signature(roi_gray, min_ink_ratio=0.004, min_span_ratio=0.35, min_area_ratio=0.0025,
                     max_ink_level=190):
    """
    Mengembalikan (present: bool, info: dict, debug_mask).

    Tanda tangan dicirikan oleh goresan tinta yang (1) cukup banyak, dan (2) membentang lebar
    sebagai satu komponen terhubung besar. Teks cetak berupa banyak komponen kecil sehingga
    tersingkir oleh filter ukuran komponen setelah operasi closing.
    """
    h, w = roi_gray.shape
    # normalisasi ukuran agar parameter morfologi konsisten antar resolusi
    scale = 400.0 / w
    g = cv2.resize(roi_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    # 1) Thresholding (Otsu pada hasil background-flattening agar kebal terhadap warna latar)
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((15, 15), np.uint8)), 21)
    flat = cv2.divide(g, bg, scale=255)
    # Otsu murni selalu "membelah" histogram, bahkan pada kertas kosong (noise -> foreground palsu).
    # Karena itu ambang dibatasi: tinta harus jauh lebih gelap dari latar (flat latar ~255).
    t_otsu, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t = min(t_otsu, max_ink_level)
    _, bw = cv2.threshold(flat, t, 255, cv2.THRESH_BINARY_INV)

    # 2) Morfologi: opening (buang bintik noise) lalu closing (sambung goresan putus)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # 3) Analisis komponen terhubung
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    H, W = bw.shape
    best = None
    mask = np.zeros_like(bw)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area_ratio * H * W:
            continue
        if best is None or cw > best[2]:
            best = (x, y, cw, ch, area)
        mask[lab == i] = 255

    ink = float(np.count_nonzero(mask)) / (H * W)
    span = best[2] / W if best else 0.0
    present = bool(best is not None and ink >= min_ink_ratio and span >= min_span_ratio)
    info = {"ink_ratio": round(ink, 4), "max_span_ratio": round(float(span), 3)}
    return present, info, mask
