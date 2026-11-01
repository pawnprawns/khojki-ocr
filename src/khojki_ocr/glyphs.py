"""Projection proposals and fixed 48-pixel input normalization."""

import cv2
import numpy as np


def runs(mask):
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def detect_lines(ink):
    H, W = ink.shape
    # Joining short vertical gaps keeps detached vowel signs with their baseline.
    projection = (ink > 0).sum(axis=1)
    smooth = np.convolve(projection, np.ones(5) / 5, mode="same")
    active = smooth > max(12, np.percentile(smooth, 90) * 0.17)
    gap = max(3, round(H * 0.006))
    joined = cv2.morphologyEx(
        active.astype(np.uint8).reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((gap, 1), np.uint8),
    ).ravel()
    boxes = []
    for top, bottom in runs(joined):
        if bottom - top < max(9, H * 0.005) or bottom - top > H * 0.12:
            continue
        ys, xs = np.where(ink[top:bottom] > 0)
        if not len(xs) or xs.max() - xs.min() < W * 0.025:
            continue
        pad = max(3, round(H * 0.002))
        boxes.append(
            list(
                map(
                    int,
                    [
                        max(0, int(xs.min()) - pad),
                        max(0, top - pad),
                        min(W, int(xs.max()) + pad + 1),
                        min(H, bottom + pad),
                    ],
                )
            )
        )
    return boxes


def glyph_boxes(line_ink, join_gap=0):
    """Projection baseline. Detached signs sharing x coordinates stay together.

    Touching glyphs and side-bearing marks remain an explicit limitation.
    """
    active = (line_ink > 0).any(axis=0).astype(np.uint8)
    if join_gap:
        active = cv2.morphologyEx(
            active.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, join_gap), np.uint8)
        ).ravel()
    boxes = []
    for left, right in runs(active):
        ys, xs = np.where(line_ink[:, left:right] > 0)
        if len(xs) < 3:
            continue
        boxes.append([int(left), int(ys.min()), int(right), int(ys.max()) + 1])
    return boxes


def normalize_glyph(ink, side=48):
    ys, xs = np.where(ink > 0)
    if not len(xs):
        return np.zeros((side, side), np.float32)
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    scale = (side - 8) / max(h, w)
    resized = cv2.resize(
        crop,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    out = np.zeros((side, side), np.float32)
    h, w = resized.shape
    y = (side - h) // 2
    x = (side - w) // 2
    out[y : y + h, x : x + w] = resized / 255.0
    return out
