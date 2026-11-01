"""Disjoint component grouping and conservative punctuation seams."""

import copy
import cv2
import numpy as np
from .model import is_consonant


def union(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def xgap(a, b):
    return max(0, a[0] - b[2], b[0] - a[2])


def make_unit(ink, box, ids=None, lab=None):
    l, t, r, b = box
    patch = ink[t:b, l:r].copy()
    if ids is not None:
        patch[~np.isin(lab[t:b, l:r], ids)] = 0
    return {
        "box": list(box),
        "ink": patch,
        "components": list(ids or []),
        "operations": [],
    }


def merge_units(units, operation):
    bb = union([u["box"] for u in units])
    l, t, r, b = bb
    im = np.zeros((b - t, r - l), np.uint8)
    for u in units:
        a, c, d, e = u["box"]
        im[c - t : e - t, a - l : d - l] = np.maximum(
            im[c - t : e - t, a - l : d - l], u["ink"]
        )
    return {
        "box": bb,
        "ink": im,
        "components": [i for u in units for i in u["components"]],
        "operations": [x for u in units for x in u["operations"]] + [operation],
    }


def tight_unit(patch, origin, operation):
    yy, xx = np.where(patch > 0)
    if not len(xx):
        return None
    l, t, r, b = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    return {
        "box": [origin[0] + l, origin[1] + t, origin[0] + r, origin[1] + b],
        "ink": patch[t:b, l:r].copy(),
        "components": [],
        "operations": [operation],
    }


def dot_geometry(box, h):
    w = box[2] - box[0]
    hh = box[3] - box[1]
    return hh <= 0.35 * h and w <= 0.4 * h and 0.5 <= w / max(1, hh) <= 1.8


def separator_geometry(im, h):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(im, 8)
    if n != 3:
        return False
    a, b = sorted(
        [[int(x), int(y), int(x + w), int(y + hh)] for x, y, w, hh, _ in stats[1:]],
        key=lambda z: z[1],
    )
    ha = a[3] - a[1]
    hb = b[3] - b[1]
    gap = (b[1] + b[3] - a[1] - a[3]) / 2
    xo = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ratio = stats[1, 4] / stats[2, 4]
    return (
        dot_geometry(a, h)
        and dot_geometry(b, h)
        and 0.15 * h <= gap <= 0.8 * h
        and xo >= 0.5 * min(a[2] - a[0], b[2] - b[0])
        and 0.4 <= ratio <= 2.5
    )


class Recognizer:
    def component_groups(self, ink):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        cc = {
            k: make_unit(ink, [int(x), int(y), int(x + w), int(y + h)], [k], lab)
            for k, (x, y, w, h, a) in enumerate(stats[1:], 1)
        }
        if not cc:
            return [], []
        heights = [u["box"][3] - u["box"][1] for u in cc.values()]
        large = [h for h in heights if h >= 0.4 * ink.shape[0]]
        H = float(np.median(large or heights))
        anchors = [
            k
            for k, u in cc.items()
            if u["box"][3] - u["box"][1] >= 0.5 * H
            and np.count_nonzero(u["ink"]) >= 0.025 * H * H
        ]
        clusters = [[k] for k in anchors]
        # A disconnected lower U can enclose its base; neighboring bodies need not merge.
        changed = True
        while changed:
            changed = False
            for i in range(len(clusters)):
                if changed:
                    break
                for j in range(i + 1, len(clusters)):
                    a = union([cc[k]["box"] for k in clusters[i]])
                    b = union([cc[k]["box"] for k in clusters[j]])
                    if overlap(a, b) >= 0.85 * min(area(a), area(b)):
                        clusters[i] += clusters.pop(j)
                        changed = True
                        break
        bodies = [
            (
                merge_units([cc[k] for k in ids], "nested_body_parts")
                if len(ids) > 1
                else copy.deepcopy(cc[ids[0]])
            )
            for ids in clusters
        ]
        hints, _ = self.classify(bodies)
        for u, p in zip(bodies, hints):
            u["anchor_label"] = p["label"]
            u["attached_marks"] = False
        used = set(anchors)
        rest = [k for k in cc if k not in used]
        punct = []
        events = []
        bodycy = (
            float(np.median([(u["box"][1] + u["box"][3]) / 2 for u in bodies]))
            if bodies
            else ink.shape[0] / 2
        )
        # Reconstruct separated colon dots before considering isolated mark attachment.
        for i in rest:
            if i in used or not dot_geometry(cc[i]["box"], H):
                continue
            candidates = []
            for j in rest:
                if j <= i or j in used or not dot_geometry(cc[j]["box"], H):
                    continue
                a, b = sorted([cc[i]["box"], cc[j]["box"]], key=lambda z: z[1])
                if not ((a[1] + a[3]) / 2 < bodycy < (b[1] + b[3]) / 2):
                    continue
                u = merge_units([cc[i], cc[j]], "paired_separator_dots")
                if separator_geometry(u["ink"], H):
                    candidates.append((u["box"][3] - u["box"][1], j, u))
            if candidates:
                _, j, u = min(candidates, key=lambda x: x[0])
                punct.append(u)
                used |= {i, j}
                events.append(
                    {
                        "kind": "dot_pair",
                        "source_boxes": [cc[i]["box"], cc[j]["box"]],
                        "box": u["box"],
                    }
                )
        for k in rest:
            if k in used:
                continue
            u = cc[k]
            a = u["box"]
            ah = a[3] - a[1]
            choices = []
            for i, body in enumerate(bodies):
                b = body["box"]
                gap = xgap(a, b)
                center = (a[1] + a[3]) / 2
                bc = (b[1] + b[3]) / 2
                vertical = max(0, b[1] - a[3], a[1] - b[3])
                xo = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                if ah <= 0.45 * H and gap <= 0.3 * H and vertical <= 0.35 * H:
                    choices.append(
                        (
                            -xo / max(1, a[2] - a[0]),
                            gap,
                            abs(
                                (a[0] + a[2]) / 2
                                - min(b[2], max(b[0], (a[0] + a[2]) / 2))
                            ),
                            abs(center - bc),
                            i,
                        )
                    )
            if choices:
                i = min(choices)[-1]
                body = bodies[i]
                v = merge_units([body, u], "attached_peripheral_mark")
                v["anchor_label"] = body["anchor_label"]
                v["attached_marks"] = True
                bodies[i] = v
                events.append(
                    {
                        "kind": "component_mark_attachment",
                        "body": body["box"],
                        "mark": a,
                        "box": v["box"],
                    }
                )
                used.add(k)
        units = bodies + punct + [cc[k] for k in rest if k not in used]
        units.sort(key=lambda u: u["box"][0])
        result = []
        for u in units:
            # Only an AA fragment with extra marks needs canonical-cluster rescue.
            if result and u.get("anchor_label") == "𑈬" and u.get("attached_marks"):
                prev = result[-1]
                hint = prev.get("anchor_label", "")
                tail = [c for c in hint[1:] if 0x1122C <= ord(c) <= 0x11233]
                if (
                    hint
                    and is_consonant(ord(hint[0]))
                    and (not tail or tail == ["𑈰"] or tail == ["𑈱"])
                    and xgap(prev["box"], u["box"]) <= 0.3 * H
                ):
                    merged = merge_units([prev, u], "base_AA_attached_marks")
                    result[-1] = merged
                    events.append(
                        {
                            "kind": "AA_mark_rescue",
                            "source_boxes": [prev["box"], u["box"]],
                            "box": merged["box"],
                        }
                    )
                    continue
            result.append(u)
        assert sum(np.count_nonzero(u["ink"]) for u in result) == np.count_nonzero(
            ink
        ), "Component grouping lost or duplicated ink"
        return result, events

    def punct_split(self, units, H):
        result = []
        events = []
        for u in units:
            im = u["ink"]
            hh, w = im.shape
            options = []
            if 0.35 * H <= w <= 1.2 * H and hh >= 0.55 * H:
                _, _, stats, _ = cv2.connectedComponentsWithStats(im, 8)
                for x, y, ww, h, area_ in stats[1:]:
                    if not dot_geometry([x, y, x + ww, y + h], H):
                        continue
                    for cut in [int(x) - 1, int(x + ww) + 1]:
                        if not 0.2 * w <= cut <= 0.8 * w:
                            continue
                        left = tight_unit(im[:, :cut], u["box"][:2], "punctuation_seam")
                        right = tight_unit(
                            im[:, cut:],
                            [u["box"][0] + cut, u["box"][1]],
                            "punctuation_seam",
                        )
                        if left is None or right is None:
                            continue
                        dot_left = separator_geometry(left["ink"], H)
                        dot_right = separator_geometry(right["ink"], H)
                        if dot_left == dot_right:
                            continue
                        bar = right if dot_left else left
                        if bar["ink"].shape[1] / bar["ink"].shape[0] > 0.65:
                            continue
                        pred, _ = self.classify([left, right])
                        expected = ["𑈺", "𑈸"] if dot_left else ["𑈸", "𑈺"]
                        if [p["label"] for p in pred] == expected and min(
                            p["score"] for p in pred
                        ) >= 0.8:
                            options.append(
                                (min(p["score"] for p in pred), [left, right], cut)
                            )
            if options:
                _, parts, cut = max(options, key=lambda x: x[0])
                assert sum(
                    np.count_nonzero(x["ink"]) for x in parts
                ) == np.count_nonzero(im)
                result.extend(parts)
                events.append(
                    {
                        "kind": "punctuation_split",
                        "box": u["box"],
                        "cut_x": u["box"][0] + cut,
                        "parts": [p["box"] for p in parts],
                    }
                )
            else:
                result.append(u)
        return result, events
