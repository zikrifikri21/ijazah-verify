"""Metode image enhancement yang dibandingkan (input & output: citra grayscale uint8)."""
import cv2
import numpy as np


def none(g):
    return g


def hist_eq(g):
    return cv2.equalizeHist(g)


def clahe(g, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


def gamma(g, gm=0.6):
    lut = np.array([(i / 255.0) ** gm * 255 for i in range(256)], np.uint8)
    return cv2.LUT(g, lut)


def contrast_stretch(g):
    lo, hi = np.percentile(g, (1, 99))
    return np.clip((g.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def median(g, k=3):
    return cv2.medianBlur(g, k)


def gaussian(g, k=3):
    return cv2.GaussianBlur(g, (k, k), 0)


def bilateral(g):
    return cv2.bilateralFilter(g, 7, 40, 40)


def unsharp(g, amount=1.5, sigma=2.0):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1 + amount, blur, -amount, 0)


def upscale(g, f=2):
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def stretch_clahe(g):
    return clahe(contrast_stretch(g))


def denoise_clahe(g):
    return clahe(median(g, 3))


def denoise_stretch_sharpen(g):
    return unsharp(contrast_stretch(median(g, 3)))


def full(g):
    """Pipeline gabungan: median -> stretch -> CLAHE -> unsharp -> upscale 2x."""
    return upscale(unsharp(clahe(contrast_stretch(median(g, 3)))))


def otsu(g):
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive(g):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)


def stretch_otsu(g):
    return otsu(contrast_stretch(median(g, 3)))


METHODS = {
    "none": none, "hist_eq": hist_eq, "clahe": clahe, "gamma": gamma,
    "contrast_stretch": contrast_stretch, "median": median, "gaussian": gaussian,
    "bilateral": bilateral, "unsharp": unsharp, "upscale2x": upscale,
    "stretch+clahe": stretch_clahe, "median+clahe": denoise_clahe,
    "median+stretch+unsharp": denoise_stretch_sharpen, "full": full,
    "otsu": otsu, "adaptive": adaptive, "median+stretch+otsu": stretch_otsu,
}

# enhancement global untuk seluruh citra (langkah 3 pada pipeline)
GLOBAL_ENHANCE = "contrast_stretch"
