"""Illumination normalization, Otsu thresholding and component filtering."""

import cv2, numpy as np


def foreground(gray, with_info=False):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    bg = cv2.GaussianBlur(gray, (0, 0), max(5, H / 80))
    flat = cv2.divide(gray, bg, scale=255)
    threshold, ink = cv2.threshold(
        flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.ones(n, bool)
    keep[0] = False
    keep &= (
        (stats[:, 4] >= max(4, H * W / 250000))
        & (stats[:, 3] <= H * 0.22)
        & ~((stats[:, 2] > W * 0.7) & (stats[:, 3] < H * 0.04))
    )
    clean = np.where(keep[lab], 255, 0).astype(np.uint8)
    info = {
        "otsu_threshold": threshold,
        "minimum_component_area": max(4, H * W / 250000),
        "raw_ink_pixels": int(np.count_nonzero(ink)),
        "retained_ink_pixels": int(np.count_nonzero(clean)),
    }
    return (clean, ink, info) if with_info else (clean, ink)
