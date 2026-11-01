"""Local body trajectories with peripheral-mark ownership for unframed layouts.

This heuristic proposes reviewable regions, not validated handwriting OCR. Long
ruling pixels are removed separately; retained marks are assigned after bodies.
"""

import cv2
import numpy as np


def components(mask):
    return cv2.connectedComponentsWithStats(mask, 8)


def remove_rules(raw):
    H, W = raw.shape
    removed = np.zeros_like(raw)
    vertical = []
    for axis, kernel in [
        ("vertical", (1, max(25, round(H * 0.075)))),
        ("horizontal", (max(35, round(W * 0.15)), 1)),
    ]:
        candidate = cv2.morphologyEx(
            raw, cv2.MORPH_OPEN, np.ones(kernel[::-1], np.uint8)
        )
        n, lab, stats, _ = components(candidate)
        for i, (x, y, w, h, a) in enumerate(stats[1:], 1):
            elongated = (
                (h > 8 * w and w < W * 0.035)
                if axis == "vertical"
                else (w > 12 * h and h < H * 0.025)
            )
            if elongated:
                removed[lab == i] = 255
                if axis == "vertical":
                    vertical.append([int(x), int(y), int(x + w), int(y + h)])
    # Group the overlapping sections of a slightly tilted internal ruling.
    groups = []
    for b in sorted(vertical, key=lambda b: (b[0] + b[2]) / 2):
        x = (b[0] + b[2]) / 2
        if (
            groups
            and abs(x - np.median([(v[0] + v[2]) / 2 for v in groups[-1]])) < W * 0.02
        ):
            groups[-1].append(b)
        else:
            groups.append([b])
    dividers = []
    for g in groups:
        x = float(np.median([(b[0] + b[2]) / 2 for b in g]))
        top = min(b[1] for b in g)
        bottom = max(b[3] for b in g)
        if W * 0.25 < x < W * 0.75 and bottom - top > H * 0.3:
            dividers.append([x, top, bottom])
    return np.where(removed, 0, raw).astype(np.uint8), removed, dividers


def body_scale(stats, H, W):
    s = stats[1:]
    ok = (
        (s[:, 3] >= max(9, H * 0.006))
        & (s[:, 3] < H * 0.08)
        & (s[:, 2] < W * 0.15)
        & (s[:, 2] >= s[:, 3] * 0.16)
        & (s[:, 2] <= s[:, 3] * 5)
    )
    s = s[ok]
    if not len(s):
        return None
    order = np.argsort(s[:, 3])
    s = s[order]
    weights = np.minimum(s[:, 4], np.percentile(s[:, 4], 85))
    return float(s[np.searchsorted(np.cumsum(weights), weights.sum() * 0.5), 3])


def fit_group(ids, stats):
    s = stats[ids]
    x = s[:, 0] + s[:, 2] * 0.5
    y = s[:, 1] + s[:, 3] * 0.75
    if len(ids) > 2 and np.ptp(x) > 1:
        slope = float(np.clip(np.polyfit(x, y, 1)[0], -0.25, 0.25))
        intercept = float(np.median(y - slope * x))
    else:
        slope = 0.0
        intercept = float(np.median(y))
    return slope, intercept


def baseline_groups(ids, stats, height):
    if not len(ids):
        return []
    ids = np.array(ids)
    s = stats[ids]
    x = s[:, 0] + s[:, 2] * 0.5
    y = s[:, 1] + s[:, 3] * 0.75
    parent = list(range(len(ids)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    # Mutual local neighbors make trajectories without a global horizontal band.
    for i in range(len(ids)):
        dx = x - x[i]
        dy = np.abs(y - y[i])
        ok = (
            (dx > 0)
            & (dx < height * 4)
            & (dy < np.minimum(0.8 * height, 0.45 * height + 0.10 * dx))
        )
        js = np.where(ok)[0]
        if len(js):
            j = int(js[np.argmin(dy[js] / height + 0.12 * dx[js] / height)])
            parent[find(j)] = find(i)
    groups = {}
    for i, k in enumerate(ids):
        groups.setdefault(find(i), []).append(int(k))
    groups = list(groups.values())
    # Join interrupted words only if both fitted trajectories agree locally.
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            a = groups[i]
            sa = stats[a]
            al = sa[:, 0].min()
            ar = (sa[:, 0] + sa[:, 2]).max()
            ma, ba = fit_group(a, stats)
            for j in range(i + 1, len(groups)):
                b = groups[j]
                sb = stats[b]
                bl = sb[:, 0].min()
                br = (sb[:, 0] + sb[:, 2]).max()
                mb, bb = fit_group(b, stats)
                gap = max(bl - ar, al - br, 0)
                if gap > height * 7:
                    continue
                mid = (max(al, bl) + min(ar, br)) / 2
                overlap = max(0, min(ar, br) - max(al, bl))
                if overlap > height * 1.5:
                    continue
                if abs((ma * mid + ba) - (mb * mid + bb)) < height * 0.5:
                    groups[i] += groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups


def component_lines(foreground, raw, old):
    H, W = raw.shape
    clean, rules, dividers = remove_rules(raw)
    if old[1]["frame"]["body_rectangle"]:
        clean = old[2].copy()
    n, lab, stats, centres = components(clean)
    area_limit = max(4, H * W / 250000)
    height = body_scale(stats, H, W)
    if height is None:
        return (
            [],
            {
                "route": "component_baselines",
                "frame": old[1]["frame"],
                "line_anchor_count": 0,
                "components": [],
                "all_text_ink_owned_once": True,
                "retained_ink_pixels": 0,
                "rejected_nontext_pixels": int(np.count_nonzero(clean)),
                "blank_or_no_supported_text": True,
            },
            np.zeros_like(raw),
            raw.copy(),
        )
    valid = np.ones(n, bool)
    valid[0] = False
    valid &= (stats[:, 4] >= area_limit) & (stats[:, 3] < max(H * 0.12, height * 4))
    bodies = np.where(
        valid
        & (stats[:, 3] >= height * 0.70)
        & (stats[:, 3] < height * 2.2)
        & (stats[:, 4] >= height * height * 0.14)
        & (stats[:, 2] >= height * 0.16)
        & (stats[:, 2] < height * 7)
    )[0]
    # Partition only the portion supported by an actual internal ruling.
    zones = {}
    zone_of = {}
    for k in bodies:
        x, y = centres[k]
        z = 0
        if dividers:
            dx, top, bottom = dividers[0]
            if top - height * 0.5 < y < bottom + height * 0.3:
                z = 1 if x < dx else 2
        zones.setdefault(z, []).append(int(k))
        zone_of[int(k)] = z
    groups = []
    group_zones = []
    for z, ids in zones.items():
        for g in baseline_groups(ids, stats, height):
            if len(g) >= 3 or (
                len(g) >= 2 and sum(stats[k, 4] for k in g) > height * height
            ):
                groups.append(g)
                group_zones.append(z)
    owners = [list(g) for g in groups]
    body_owner = {k: i for i, g in enumerate(groups) for k in g}
    remaining = [int(k) for k in np.where(valid)[0] if int(k) not in body_owner]
    for k in remaining:
        x, y, w, h, a = map(float, stats[k])
        cx, cy = centres[k]
        options = []
        for i, g in enumerate(groups):
            z = group_zones[i]
            if dividers and z:
                dx, top, bottom = dividers[0]
                if (z == 1 and cx > dx) or (z == 2 and cx < dx):
                    continue
            for b in g:
                bx, by, bw, bh, _ = stats[b]
                gx = max(bx - (x + w), x - (bx + bw), 0)
                gy = max(by - (y + h), y - (by + bh), 0)
                # Prefer nearby, x-overlapping bodies for peripheral signs.
                score = (
                    (gx / height) ** 2
                    + (gy / height) ** 2
                    + 0.08 * abs(cy - (by + bh * 0.55)) / height
                )
                if gx < height * 1.4 and gy < height * 0.75:
                    options.append((score, i))
        if options:
            owners[min(options)[1]].append(k)
    text = np.zeros_like(raw)
    count = np.zeros_like(raw, np.uint16)
    lines = []
    records = []
    for j, ids in enumerate(owners):
        s = stats[ids]
        x = max(0, int(s[:, 0].min()) - 2)
        y = max(0, int(s[:, 1].min()) - 2)
        r = min(W, int((s[:, 0] + s[:, 2]).max()) + 2)
        b = min(H, int((s[:, 1] + s[:, 3]).max()) + 2)
        keep = np.zeros(n, bool)
        keep[ids] = True
        mask = np.where(keep[lab[y:b, x:r]], clean[y:b, x:r], 0).astype(np.uint8)
        text[y:b, x:r] |= mask
        count[y:b, x:r] += (mask > 0).astype(np.uint16)
        slope, intercept = fit_group(groups[j], stats)
        warnings = ["adaptive_layout_requires_review"]
        if b - y > height * 2.5:
            warnings.append("possible_touching_rows_review")
        if len(groups[j]) < 4:
            warnings.append("sparse_text_region_review")
        lines.append(
            {
                "bbox": [x, y, r, b],
                "ink": mask,
                "component_ids": ids,
                "anchor_bbox": [x, y, r, b],
                "warnings": warnings,
                "baseline": [slope, intercept],
                "block": group_zones[j],
                "_sort_y": slope * ((x + r) / 2) + intercept,
            }
        )
        records += [
            {
                "component": k,
                "box": [
                    int(stats[k, 0]),
                    int(stats[k, 1]),
                    int(stats[k, 0] + stats[k, 2]),
                    int(stats[k, 1] + stats[k, 3]),
                ],
                "anchor": j,
            }
            for k in ids
        ]
    assert np.all(count[text > 0] == 1)
    # Physical row ordering uses the fitted body trajectory, never a tall rule.
    lines.sort(key=lambda l: (l["_sort_y"], l["bbox"][0]))
    for i, l in enumerate(lines, 1):
        l["id"] = f"line-{i:03}"
        l.pop("_sort_y")
    removed = np.where((raw > 0) & (text == 0), raw, 0).astype(np.uint8)
    audit = {
        "route": "component_baselines",
        "frame": old[1]["frame"],
        "components": records,
        "line_anchor_count": len(groups),
        "nominal_band_height": height,
        "all_text_ink_owned_once": True,
        "retained_ink_pixels": int(np.count_nonzero(text)),
        "removed_border_pixels": int(np.count_nonzero(rules)),
        "rejected_nontext_pixels": int(np.count_nonzero(removed)),
        "internal_dividers": dividers,
        "input_plane": (
            "framed_foreground"
            if old[1]["frame"]["body_rectangle"]
            else "threshold_raw"
        ),
        "blank_or_no_supported_text": not len(lines),
    }
    return lines, audit, text, removed
