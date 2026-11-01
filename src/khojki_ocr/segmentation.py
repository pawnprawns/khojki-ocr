"""Current-model guidance, complete marks, and fixed structural grouping."""

import copy, hashlib
import numpy as np
from .glyphs import normalize_glyph, glyph_boxes
from .model import is_consonant
from .component_grouping import Recognizer, make_unit, merge_units, xgap
from .geometry import (
    cc,
    tight,
    canvas,
    dot,
    separator,
    bar,
    bar_pair,
    punctuation,
    SEP,
    DD,
    AN,
)

DANDA = "\U00011238"


def body_height(ink):
    _, parts = cc(ink)
    hs = [c["h"] for c in parts]
    if not hs:
        return float(ink.shape[0])
    q = float(np.percentile(hs, 90))
    return float(np.median([h for h in hs if 0.5 * q <= h <= 2.2 * q]))


class CurrentCropper(Recognizer):
    def __init__(self, predictor):
        self.predictor = predictor
        self.cache = {}
        self.context_H = 43.0
        self.body_fix = False
        self.small_fix = False

    def classify(self, units):
        images = [np.uint8(np.rint(normalize_glyph(u["ink"]) * 255)) for u in units]
        keys = [hashlib.sha256(im.tobytes()).hexdigest() for im in images]
        missing = {k: im for k, im in zip(keys, images) if k not in self.cache}
        if missing:
            records = self.predictor.predict(np.stack(list(missing.values())))
            for key, p in zip(missing, records):
                self.cache[key] = {**p, "score": p["top3"][0]["score"]}
        return [self.cache[k] for k in keys], images

    def merge_body_parts(self, parts, whole_height):
        parts = copy.deepcopy(parts)
        events = []
        if self.small_fix:
            # A pair of narrow, overlapping body fragments can form a complete
            # base. Require both to be too narrow to count as a full body and
            # avoid tall vowel strokes and short peripheral marks.
            changed = True
            H = self.context_H
            while changed:
                changed = False
                for i, a in enumerate(parts):
                    if changed:
                        break
                    aw, ah = a["ink"].shape[1], a["ink"].shape[0]
                    if not (aw < 0.55 * H and 0.5 * H <= ah <= 1.35 * H):
                        continue
                    for j in range(i + 1, len(parts)):
                        b = parts[j]
                        bw, bh = b["ink"].shape[1], b["ink"].shape[0]
                        if not (bw < 0.55 * H and 0.5 * H <= bh <= 1.35 * H):
                            continue
                        A, B = a["box"], b["box"]
                        xo = min(A[2], B[2]) - max(A[0], B[0])
                        yo = min(A[3], B[3]) - max(A[1], B[1])
                        if (
                            xo > 0
                            and yo >= 0.5 * min(ah, bh)
                            and abs((A[1] + A[3] - B[1] - B[3]) / 2) <= 0.35 * H
                        ):
                            joined = merge_units(
                                [a, b], "small_overlapping_body_fragments"
                            )
                            parts[i] = joined
                            parts.pop(j)
                            events.append(
                                {
                                    "kind": "small_overlapping_body_fragments",
                                    "source_boxes": [A, B],
                                    "box": joined["box"],
                                }
                            )
                            changed = True
                            break
        if self.body_fix:
            changed = True
            while changed:
                changed = False
                hs = [
                    p["ink"].shape[0]
                    for p in parts
                    if p["ink"].shape[0] >= 0.5 * whole_height
                ]
                H = float(np.median(hs)) if hs else whole_height
                ids = [
                    i
                    for i, p in enumerate(parts)
                    if p["ink"].shape[0] >= 0.5 * H and p["ink"].shape[1] >= 0.55 * H
                ]
                for ii, i in enumerate(ids):
                    if changed:
                        break
                    for j in ids[ii + 1 :]:
                        a, b = parts[i], parts[j]
                        A, B = a["box"], b["box"]
                        ov = min(A[2], B[2]) - max(A[0], B[0])
                        if ov > 0.15 * min(a["ink"].shape[1], b["ink"].shape[1]):
                            joined = merge_units(
                                [a, b], "overlapping_substantial_body_parts"
                            )
                            parts[i] = joined
                            parts.pop(j)
                            events.append(
                                {
                                    "kind": "overlapping_substantial_body_parts",
                                    "source_boxes": [A, B],
                                    "box": joined["box"],
                                }
                            )
                            changed = True
                            break
        return sorted(parts, key=lambda p: p["box"][0]), events

    def component_groups(self, ink):
        original = [make_unit(ink, bb) for bb in glyph_boxes(ink)]
        result = []
        events = []
        for u in original:
            parts, ev = Recognizer.component_groups(self, u["ink"])
            parts, merge_events = self.merge_body_parts(parts, u["ink"].shape[0])
            heights = [
                p["ink"].shape[0]
                for p in parts
                if p["ink"].shape[0] >= 0.5 * u["ink"].shape[0]
            ]
            H = float(np.median(heights)) if heights else float(u["ink"].shape[0])
            substantial = [
                p
                for p in parts
                if p["ink"].shape[0] >= 0.5 * H and p["ink"].shape[1] >= 0.55 * H
            ]
            ordered = sorted(substantial, key=lambda p: p["box"][0])
            distinct = len(ordered) >= 2 and all(
                max(0, a["box"][2] - b["box"][0])
                <= 0.15 * min(a["ink"].shape[1], b["ink"].shape[1])
                for a, b in zip(ordered, ordered[1:])
            )
            accept = distinct and u["ink"].shape[1] >= 1.4 * H
            if accept:
                moved = []
                for p in parts:
                    p = copy.deepcopy(p)
                    p["box"] = [
                        p["box"][0] + u["box"][0],
                        p["box"][1] + u["box"][1],
                        p["box"][2] + u["box"][0],
                        p["box"][3] + u["box"][1],
                    ]
                    p["operations"].append("a14_multi_body_split")
                    moved.append(p)
                result.extend(moved)
                events.append(
                    {
                        "kind": "a14_multi_body_split",
                        "box": u["box"],
                        "parts": [p["box"] for p in moved],
                        "part_merges": merge_events,
                    }
                )
            else:
                if len(parts) == 1:
                    for key in ["anchor_label", "attached_marks"]:
                        if key in parts[0]:
                            u[key] = parts[0][key]
                result.append(u)
        hints, _ = self.classify(result)
        out = []
        heights = [u["ink"].shape[0] for u in result]
        H = float(np.median(heights)) if heights else ink.shape[0]
        for u, p in zip(result, hints):
            if out and u.get("anchor_label") == "𑈬" and u.get("attached_marks"):
                prev = out[-1]
                hint = prev.get("full_label", "")
                vowels = [c for c in hint[1:] if 0x1122C <= ord(c) <= 0x11233]
                if (
                    hint
                    and is_consonant(ord(hint[0]))
                    and (not vowels or vowels == ["𑈰"] or vowels == ["𑈱"])
                    and xgap(prev["box"], u["box"]) <= 0.3 * H
                ):
                    q = merge_units([prev, u], "selective_AA_mark_rescue")
                    out[-1] = q
                    events.append(
                        {
                            "kind": "selective_AA_mark_rescue",
                            "source_boxes": [prev["box"], u["box"]],
                            "box": q["box"],
                        }
                    )
                    continue
            u["full_label"] = p["label"]
            out.append(u)
        assert sum(np.count_nonzero(u["ink"]) for u in out) == sum(
            np.count_nonzero(u["ink"]) for u in original
        )
        return out, events


def convert(u):
    return {
        "box": u["box"][:],
        "ink": u["ink"].copy(),
        "parents": [],
        "events": u.get("operations", [])[:],
        "forced": None,
    }


def single_bar(u, H):
    _, parts = cc(u["ink"])
    if len(parts) != 1 or not bar(parts[0], H):
        return None
    c = parts[0].copy()
    c["x"] += u["box"][0]
    c["y"] += u["box"][1]
    return c


def combine_two(a, b, kind, forced):
    x0 = min(a["box"][0], b["box"][0])
    y0 = min(a["box"][1], b["box"][1])
    x1 = max(a["box"][2], b["box"][2])
    y1 = max(a["box"][3], b["box"][3])
    im = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for u in [a, b]:
        x, y, r, z = u["box"]
        im[y - y0 : z - y0, x - x0 : r - x0] |= u["ink"]
    return tight(im, (x0, y0), events=a["events"] + b["events"] + [kind], forced=forced)


def numeric_punctuation_seam(u, H):
    _, parts = cc(u["ink"])
    cuts = set()
    for c in parts:
        if dot(c, H):
            cuts.update([c["x"] - 1, c["x"], c["x"] + 1])
    options = []
    for cut in sorted(cuts):
        if not 0 < cut < u["ink"].shape[1]:
            continue
        left = tight(
            u["ink"][:, :cut],
            u["box"][:2],
            events=u["events"] + ["numeric_punctuation_seam"],
        )
        right = tight(
            u["ink"][:, cut:],
            (u["box"][0] + cut, u["box"][1]),
            events=u["events"] + ["numeric_punctuation_seam"],
        )
        if left is None or right is None:
            continue
        _, lc = cc(left["ink"])
        _, rc = cc(right["ink"])
        if not separator(rc, H):
            continue
        label = (
            DANDA
            if len(lc) == 1 and bar(lc[0], H)
            else DD if len(lc) == 2 and bar_pair(*lc, H) else None
        )
        if label:
            left["forced"] = label
            right["forced"] = SEP
            options.append(
                (
                    int(np.count_nonzero(u["ink"][:, cut - 1 : cut + 1])),
                    cut,
                    [left, right],
                )
            )
    return min(options, key=lambda x: (x[0], x[1]))[2] if options else [u]


def punctuate(units, H):
    # The broad v1 two-bar-suffix rule confused GA/YA plus AA with punctuation.
    # Restrict suffix peeling and connected punctuation seams to a number span
    # opened by a double danda at a word boundary. No digit value is assumed.
    initial = []
    for u in units:
        _, parts = cc(u["ink"])
        split = bool(len(parts) == 3 and all(dot(c, H) for c in parts))
        initial.extend(punctuation(u, H, split))
    out = []
    inside = False
    count = 0
    i = 0
    while i < len(initial):
        u = initial[i]
        at_boundary = not out or out[-1].get("forced") == SEP
        a = single_bar(u, H)
        b = single_bar(initial[i + 1], H) if i + 1 < len(initial) else None
        if a and b and (at_boundary or inside) and bar_pair(a, b, H):
            u = combine_two(u, initial[i + 1], "boundary_double_danda", DD)
            i += 1
        parts = [u]
        if inside and u.get("forced") is None:
            parts = punctuation(u, H, True)
            if len(parts) == 1:
                parts = numeric_punctuation_seam(u, H)
        for p in parts:
            if p["forced"] == SEP:
                inside = False
                count = 0
            elif p["forced"] == DD:
                if inside:
                    inside = False
                elif not out or out[-1].get("forced") == SEP:
                    inside = True
                    count = 0
            elif inside:
                count += 1
                if count > 4:
                    inside = False
            out.append(p)
        i += 1
    return out


def segment_line(ink, cropper, context_H=None):
    H = body_height(ink) if context_H is None else float(context_H)
    cropper.context_H = H
    cropper.body_fix = True
    cropper.small_fix = True
    units, events = cropper.component_groups(ink)
    # Input callers supply an already-cleaned foreground line. Keep all its
    # pixels; do not apply the old page-level noise filter a second time.
    occupied = canvas([convert(u) for u in units], ink.shape)
    residual = np.where((ink > 0) & (occupied == 0), ink, 0).astype(np.uint8)
    lab, missing = cc(residual)
    for c in missing:
        u = make_unit(
            residual, [c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]], [c["k"]], lab
        )
        u["operations"].append("retain_isolated_projection_speck")
        units.append(u)
    units.sort(key=lambda u: (u["box"][0], u["box"][1]))
    legacy_H = (
        H
        if context_H is not None
        else float(np.median([u["ink"].shape[0] for u in units])) if units else H
    )
    units, ev = cropper.punct_split(units, legacy_H)
    events += ev
    result = [convert(u) for u in units]
    result = punctuate(result, H)
    result.sort(key=lambda u: (u["box"][0], u["box"][1]))
    assert np.array_equal(
        canvas(result, ink.shape), ink
    ), "Lost or duplicated foreground"
    return result, events, H
