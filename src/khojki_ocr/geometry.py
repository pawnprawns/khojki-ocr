"""Frozen geometry predicates; proportions are empirical, not script guarantees."""

import copy
import cv2
import numpy as np

SEP = "\U0001123a"
DD = "\U00011239"
AN = "\U00011234"


def tight(mask, origin=(0, 0), parents=(), events=(), forced=None):
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    return {
        "box": [origin[0] + x0, origin[1] + y0, origin[0] + x1, origin[1] + y1],
        "ink": mask[y0:y1, x0:x1].copy(),
        "parents": list(parents),
        "events": list(events),
        "forced": forced,
    }


def cc(mask):
    n, lab, s, _ = cv2.connectedComponentsWithStats(mask, 8)
    return lab, [
        {"k": k, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(a)}
        for k, (x, y, w, h, a) in enumerate(s[1:], 1)
    ]


def dot(c, H):
    return (
        0.10 * H <= min(c["w"], c["h"])
        and max(c["w"], c["h"]) <= 0.50 * H
        and 0.50 <= c["w"] / c["h"] <= 1.8
        and c["area"] >= 0.015 * H * H
    )


def separator(parts, H):
    if len(parts) != 2 or not all(dot(c, H) for c in parts):
        return False
    a, b = sorted(parts, key=lambda c: c["y"])
    dy = b["y"] + b["h"] / 2 - a["y"] - a["h"] / 2
    overlap = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    return (
        0.25 * H <= dy <= 0.85 * H
        and overlap >= 0.5 * min(a["w"], b["w"])
        and 0.4 <= a["area"] / b["area"] <= 2.5
    )


def bar(c, H):
    return (
        0.65 * H <= c["h"] <= 1.35 * H
        and 0.12 * H <= c["w"] <= 0.58 * H
        and c["w"] / c["h"] <= 0.55
        and 0.10 * H * H <= c["area"] <= 0.60 * H * H
    )


def bar_pair(a, b, H):
    a, b = sorted([a, b], key=lambda c: c["x"])
    return (
        bar(a, H)
        and bar(b, H)
        and -0.08 * H <= b["x"] - (a["x"] + a["w"]) <= 0.22 * H
        and abs(a["y"] - b["y"]) <= 0.18 * H
        and abs(a["y"] + a["h"] - b["y"] - b["h"]) <= 0.18 * H
    )


def part(u, lab, ids, kind, forced=None):
    im = u["ink"].copy()
    im[~np.isin(lab, ids)] = 0
    return tight(im, u["box"][:2], u["parents"], u["events"] + [kind], forced)


def punctuation(u, H, split):
    lab, parts = cc(u["ink"])
    if separator(parts, H):
        v = copy.deepcopy(u)
        v["forced"] = SEP
        v["events"].append("native_separator")
        return [v]
    if len(parts) == 2 and bar_pair(*parts, H):
        v = copy.deepcopy(u)
        v["forced"] = DD
        v["events"].append("native_double_danda")
        return [v]
    if not split:
        return [u]
    if len(parts) == 3 and all(dot(c, H) for c in parts):
        a, b, c = sorted(parts, key=lambda c: c["y"])
        if (
            separator([b, c], H)
            and a["y"] + a["h"] <= b["y"] + 0.10 * H
            and a["x"] + a["w"] / 2 <= min(b["x"], c["x"]) + 0.25 * H
            and u["ink"].shape[0] >= 0.7 * H
        ):
            return [
                part(u, lab, [a["k"]], "upper_nasal_plus_separator", AN),
                part(u, lab, [b["k"], c["k"]], "upper_nasal_plus_separator", SEP),
            ]
    # Separate an intact two-bar suffix from preceding body components. This
    # works even when their x projections overlap; masks remain disjoint.
    ordered = sorted(parts, key=lambda c: c["x"])
    if len(parts) >= 3:
        a, b = ordered[-2:]
        rest = ordered[:-2]
        if (
            bar_pair(a, b, H)
            and all(c["x"] + c["w"] / 2 < a["x"] + 0.15 * H for c in rest)
            and any(c["w"] >= 0.60 * H and c["h"] >= 0.50 * H for c in rest)
        ):
            left = part(u, lab, [c["k"] for c in rest], "separate_two_bar_suffix")
            right = part(u, lab, [a["k"], b["k"]], "separate_two_bar_suffix", DD)
            return [left, right]
    return [u]


def canvas(units, shape):
    im = np.zeros(shape, np.uint8)
    for u in units:
        x0, y0, x1, y1 = u["box"]
        target = im[y0:y1, x0:x1]
        assert not np.any(
            (target > 0) & (u["ink"] > 0)
        ), "Duplicated foreground ownership"
        target[:] = np.maximum(target, u["ink"])
    return im
