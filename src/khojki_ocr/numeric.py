"""Cautious digit decoding reuses existing class probabilities; no CNN call."""

import copy
import numpy as np
from . import assembly
from .geometry import DD, SEP

DIGITS = [chr(0xAE6 + i) for i in range(10)]


def decode(meta, predictions, options, mode="hard"):
    assert mode in ["hard", "cautious"]
    out = copy.deepcopy(predictions)
    events = []
    for line in meta["lines"]:
        ids = line["input_indices"]
        tokens = [predictions[i]["label"] for i in ids]
        virtual, _ = assembly.assemble_dandas(tokens, line["glyph_boxes"], ids)
        seq = [(i, t) for i, t in zip(ids, virtual) if t]
        H = float(
            line.get(
                "geometry_H",
                np.median(
                    [
                        meta["units"][i]["box"][3] - meta["units"][i]["box"][1]
                        for i in ids
                    ]
                ),
            )
        )
        j = 0
        while j < len(seq):
            if seq[j][1] != DD or (j > 0 and seq[j - 1][1] != SEP):
                j += 1
                continue
            k = j + 1
            while k < len(seq) and k - j <= 5 and seq[k][1] not in [SEP, DD]:
                k += 1
            inside = seq[j + 1 : k]
            if (
                k >= len(seq)
                or seq[k][1] != DD
                or not 1 <= len(inside) <= 4
                or not any(t in DIGITS for _, t in inside)
            ):
                j += 1
                continue
            eligible = True
            for i, t in inside:
                x, y, r, b = meta["units"][i]["box"]
                if not (0.55 * H <= b - y <= 1.4 * H and 0.10 * H <= r - x <= 1.2 * H):
                    eligible = False
            if not eligible:
                j = k + 1
                continue
            confident = mode == "hard" or all(
                options[i]["mass"] is not None and options[i]["mass"] >= 0.01
                for i, _ in inside
            )
            event = {
                "line_id": line["id"],
                "input_indices": [i for i, _ in inside],
                "before": [t for _, t in inside],
                "mode": mode,
                "digit_options": [options[i] for i, _ in inside],
                "action": (
                    "constrain_to_digit_classes"
                    if confident
                    else "abstain_low_digit_mass"
                ),
            }
            if confident:
                for i, _ in inside:
                    out[i]["label"] = options[i]["label"]
            event["after"] = [out[i]["label"] for i, _ in inside]
            events.append(event)
            j = k + 1
    return out, events
