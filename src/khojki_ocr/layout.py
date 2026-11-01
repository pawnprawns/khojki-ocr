"""Geometry-only page-to-line assignment; never reads evaluation labels."""

import cv2, numpy as np
from .glyphs import detect_lines


def frame_components(ink, raw):
    """Infer the framed text interior and keep whole components touching it.

    Raw binary pixels locate the frame even when initial CC filtering removed
    its long sides. Clean pixels locate whitespace between frame and writing.
    No component intersecting the interior is shaved at the rectangle edge.
    The rectangle is used only if both side frames AND a broad top ornament
    are detected. Otherwise all foreground is retained for review.
    """
    H, W = ink.shape
    vertical = (raw > 0).sum(0)
    clean_vertical = (ink > 0).sum(0)
    cuts = []
    for lo, hi, side in [
        (int(0.02 * W), int(0.15 * W), "left"),
        (int(0.85 * W), int(0.98 * W), "right"),
    ]:
        peak = lo + int(np.argmax(vertical[lo:hi]))
        if vertical[peak] < 0.30 * H:
            continue
        active = (vertical[lo:hi] > 0.30 * H).astype(np.uint8)[None, :]
        active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0]
        a = b = peak - lo
        while a > 0 and active[a - 1]:
            a -= 1
        while b + 1 < len(active) and active[b + 1]:
            b += 1
        a += lo
        b += lo + 1
        if side == "left":
            glo = b
            ghi = min(W, b + round(0.04 * W))
        else:
            glo = max(0, a - round(0.04 * W))
            ghi = a
        if ghi <= glo:
            continue
        gutter = glo + int(np.argmin(clean_vertical[glo:ghi]))
        row = ((raw[:, a:b] > 0).sum(1) > max(1, 0.1 * (b - a))).astype(np.uint8)[
            :, None
        ]
        row = cv2.morphologyEx(
            row, cv2.MORPH_CLOSE, np.ones((max(11, round(0.01 * H)), 1), np.uint8)
        )[:, 0]
        edges = np.diff(np.pad(row.astype(int), (1, 1)))
        runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
        start, end = max(runs, key=lambda z: z[1] - z[0])
        cuts.append(
            {
                "side": side,
                "dense_band": [a, b],
                "gutter": gutter,
                "vertical_span": [int(start), int(end)],
            }
        )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    remove = np.zeros(n, bool)
    events = []
    ambiguous = []
    bounds = None
    top_candidates = []
    bottom_candidates = []
    if len(cuts) == 2:
        left, right = [c["gutter"] for c in cuts]
        top = min(c["vertical_span"][0] for c in cuts)
        end = max(c["vertical_span"][1] for c in cuts)
        for k, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if w < 0.12 * W or h > 0.04 * H:
                continue
            if top - 0.01 * H <= y <= top + 0.02 * H:
                top_candidates.append([int(x), int(y), int(x + w), int(y + h)])
            if y > 0.8 * H and y + h >= end - 0.035 * H:
                bottom_candidates.append([int(x), int(y), int(x + w), int(y + h)])
        if top_candidates and right - left > 0.65 * W:
            y0 = max(b[3] for b in top_candidates)
            y1 = min(b[1] for b in bottom_candidates) if bottom_candidates else end
            if y1 - y0 > 0.6 * H:
                bounds = [left, y0, right, y1]
    if bounds:
        x0, y0, x1, y1 = bounds
        inside_count = np.bincount(labels[y0:y1, x0:x1].ravel(), minlength=n)
        inside = inside_count > 0
        for k, (x, y, w, h, area) in enumerate(stats[1:], 1):
            fraction = float(inside_count[k] / area)
            large_crossing = (w > 0.04 * W or h > 0.04 * H) and fraction < 0.1
            if not inside[k] or large_crossing:
                reason = (
                    "whole_component_outside_inferred_frame_interior"
                    if not inside[k]
                    else "large_frame_component_with_less_than_10_percent_ink_inside"
                )
                remove[k] = True
                events.append(
                    {
                        "component": k,
                        "box": [int(x), int(y), int(x + w), int(y + h)],
                        "area": int(area),
                        "interior_ink_fraction": fraction,
                        "reason": reason,
                    }
                )
            elif x < x0 or y < y0 or x + w > x1 or y + h > y1:
                ambiguous.append(
                    {
                        "component": k,
                        "box": [int(x), int(y), int(x + w), int(y + h)],
                        "reason": "crosses_interior_boundary_retained_whole",
                    }
                )
    removed = np.where(remove[labels], ink, 0).astype(np.uint8)
    kept = np.where(remove[labels], 0, ink).astype(np.uint8)
    assert np.array_equal(np.bitwise_or(kept, removed), ink)
    return (
        kept,
        removed,
        {
            "dense_columns": cuts,
            "body_rectangle": bounds,
            "top_ornament_evidence": top_candidates,
            "bottom_ornament_evidence": bottom_candidates,
            "removed_components": events,
            "retained_ambiguous_components": ambiguous,
            "fallback_all_foreground": bounds is None,
        },
    )


def projection_lines(foreground, raw=None):
    text, border, frame = frame_components(
        foreground, foreground if raw is None else raw
    )
    n, labels, stats, centres = cv2.connectedComponentsWithStats(text, 8)
    anchors = detect_lines(text)
    H, W = text.shape
    if not n > 1:
        return (
            [],
            {"frame": frame, "components": [], "all_text_ink_owned_once": True},
            text,
            border,
        )
    if not anchors:
        ys, xs = np.where(text)
        anchors = [[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]]
    nominal = float(np.median([b[3] - b[1] for b in anchors]))
    owners = [[] for _ in anchors]
    orphans = []
    records = []
    for k in range(1, n):
        x, y, w, h, area = map(int, stats[k])
        bottom = y + h
        scores = []
        for j, (_, top, _, end) in enumerate(anchors):
            overlap = max(0, min(bottom, end) - max(y, top))
            gap = max(top - bottom, y - end, 0)
            score = (-overlap / max(h, 1), gap, abs((y + bottom - top - end) / 2))
            scores.append((score, j))
        score, j = min(scores)
        if score[1] > 0.65 * nominal:
            orphans.append(k)
            owner = None
        else:
            owners[j].append(k)
            owner = j
        records.append(
            {
                "component": k,
                "box": [x, y, x + w, bottom],
                "area": area,
                "anchor": owner,
                "gap": score[1],
                "small_mark_candidate": h < 0.35 * nominal and w < 0.45 * nominal,
            }
        )
    # Keep distant isolated content in separate review regions, not a nearby line.
    for k in sorted(orphans, key=lambda k: centres[k, 1]):
        x, y, w, h, _ = map(int, stats[k])
        candidates = [
            j
            for j in range(len(owners))
            if j >= len(anchors)
            and abs(centres[owners[j][0], 1] - centres[k, 1]) < 0.3 * nominal
        ]
        if candidates:
            owners[candidates[0]].append(k)
        else:
            owners.append([k])
    lines = []
    global_owners = np.full(n, -1, np.int32)
    for j, ids in enumerate(owners):
        if not ids:
            continue
        boxes = stats[ids]
        x = max(0, int(boxes[:, 0].min()) - 2)
        y = max(0, int(boxes[:, 1].min()) - 2)
        r = min(W, int((boxes[:, 0] + boxes[:, 2]).max()) + 2)
        b = min(H, int((boxes[:, 1] + boxes[:, 3]).max()) + 2)
        keep = np.zeros(n, bool)
        keep[ids] = True
        mask = np.where(keep[labels[y:b, x:r]], text[y:b, x:r], 0).astype(np.uint8)
        global_owners[ids] = len(lines)
        warnings = []
        if j >= len(anchors):
            warnings.append("isolated_content_outside_main_line_bands")
        if b - y < 0.6 * nominal:
            warnings.append("small_text_or_decoration_review")
        if b - y > 1.7 * nominal:
            warnings.append("unusually_tall_region_review")
        bounds = frame["body_rectangle"]
        if bounds and (y < bounds[1] + 0.025 * H or b > bounds[3] - 0.025 * H):
            warnings.append("near_frame_edge_review")
        if frame["fallback_all_foreground"]:
            warnings.append("no_reliable_frame_roi_review")
        lines.append(
            {
                "bbox": [x, y, r, b],
                "ink": mask,
                "component_ids": ids,
                "anchor_bbox": anchors[j] if j < len(anchors) else None,
                "warnings": warnings,
            }
        )
    assert (global_owners[1:] >= 0).all()
    # Each foreground connected component has one owner and remains whole.
    canvas = np.zeros_like(text)
    count = np.zeros_like(text, dtype=np.uint8)
    for line in lines:
        x, y, r, b = line["bbox"]
        canvas[y:b, x:r] |= line["ink"]
        count[y:b, x:r] += (line["ink"] > 0).astype(np.uint8)
    assert np.array_equal(canvas, text) and np.all(count[text > 0] == 1)
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    for i, l in enumerate(lines, 1):
        l["id"] = f"line-{i:03}"
    audit = {
        "frame": frame,
        "components": records,
        "line_anchor_count": len(anchors),
        "nominal_band_height": nominal,
        "all_text_ink_owned_once": True,
        "retained_ink_pixels": int(np.count_nonzero(text)),
        "removed_border_pixels": int(np.count_nonzero(border)),
    }
    return lines, audit, text, border


def find_lines(foreground, raw=None):
    """Use supported frame geometry; otherwise estimate local body trajectories.

    Routing uses image evidence only. Evaluation IDs, references and page numbers
    are not available to either detector.
    """
    raw = foreground if raw is None else raw
    result = projection_lines(foreground, raw)
    if (
        result[1]["frame"]["body_rectangle"]
        and result[1].get("line_anchor_count", 0) >= 3
    ):
        result[1]["route"] = "supported_framed_projection"
        result[1]["layout_review_required"] = False
        return result
    from .adaptive_layout import component_lines

    result = component_lines(foreground, raw, result)
    result[1]["layout_review_required"] = True
    return result
