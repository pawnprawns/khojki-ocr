"""One trained multi-head character CNN per document; inference never accepts reference text."""

import time, copy
import cv2, numpy as np, torch
from .model import load_model
from . import assembly
from .marks import apply
from .segmentation import CurrentCropper, segment_line
from .numeric import DIGITS, decode
from .glyphs import glyph_boxes
from .geometry import cc
from .trace import trace_guidance, pack


def configure():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)


class ViewAdapter:
    def __init__(self, engine):
        self.engine = engine

    def predict(self, images):
        return self.engine.classify(images, seven=False)[0]


class Engine:
    def __init__(self):
        start = time.perf_counter()
        self.model, self.schema, self.transforms, self.manifest = load_model()
        self.digit_ids = [self.schema.labels.index(d) for d in DIGITS]
        self.model_loads = 1
        self.forwards = {"guidance": 0, "final": 0}
        self.init_seconds = time.perf_counter() - start

    @torch.no_grad()
    def classify(self, images, seven=True, keep_views=False):
        assert (
            images.dtype == np.uint8
            and images.ndim == 3
            and images.shape[1:] == (48, 48)
        )
        if not len(images):
            return [], np.empty((0, len(self.schema.labels)), np.float32), None
        total = None
        views = []
        transforms = self.transforms if seven else self.transforms[:1]
        for transform in transforms:
            a = np.asarray(transform["affine"], np.float32)
            warped = []
            for im in images:
                small = cv2.warpAffine(im, a, (48, 48), flags=cv2.INTER_LINEAR)
                padded = a.copy()
                padded[:, 2] += 8
                big = cv2.warpAffine(im, padded, (64, 64), flags=cv2.INTER_LINEAR)
                assert np.array_equal(small, big[8:56, 8:56])
                big[8:56, 8:56] = 0
                if big.any():
                    raise ValueError("Transformed crop would clip ink")
                warped.append(small)
            warped = np.stack(warped)
            if keep_views:
                views.append(warped)
            chunks = []
            for begin in range(0, len(warped), 256):
                x = (
                    torch.from_numpy(warped[begin : begin + 256]).unsqueeze(1).float()
                    / 255
                )
                chunks.append(torch.softmax(self.schema.scores(self.model(x)), 1))
                self.forwards["final" if seven else "guidance"] += 1
            p = torch.cat(chunks)
            total = p if total is None else total + p
        total /= len(transforms)
        indices = torch.argsort(total, dim=1, descending=True, stable=True)[:, :3]
        records = [
            {
                "label": self.schema.labels[int(row[0])],
                "top3": [
                    {"label": self.schema.labels[int(j)], "score": float(total[i, j])}
                    for j in row
                ],
            }
            for i, row in enumerate(indices)
        ]
        return records, total.numpy().copy(), np.stack(views) if keep_views else None

    def digit_options(self, probabilities):
        rows = probabilities[:, self.digit_ids]
        return [
            {
                "label": DIGITS[int(np.argmax(p))],
                "mass": float(p.sum()),
                "score": float(np.max(p)),
                "conditional_score": float(np.max(p) / p.sum()) if p.sum() > 0 else 0.0,
                "scores": p.tolist(),
                "score_kind": "seven-view probability mass over the ten digit classes; uncalibrated",
            }
            for p in rows
        ]

    def predict_line(self, ink, trace=False):
        assert (
            ink.dtype == np.uint8
            and ink.ndim == 2
            and set(np.unique(ink)).issubset({0, 255})
        )
        begin = time.perf_counter()
        times = {}
        before = self.forwards.copy()
        context_H = None
        runtime_warnings = []
        if not ink.any():
            return (
                {
                    "prediction": "",
                    "status": "empty",
                    "timing": {"line_total": time.perf_counter() - begin},
                    "numeric_extra_forwards": 0,
                    "cnn_forwards": {"guidance": 0, "final": 0},
                },
                None,
                None,
            )
        # A photograph/noise field can produce hundreds of fragments in one
        # projection group and make the inherited pairwise grouping explode.
        # Keep the input for review instead of pretending it was recognized.
        for box in glyph_boxes(ink):
            x, y, r, b = box
            n = cv2.connectedComponents(ink[y:b, x:r], 8)[0] - 1
            if n > 128:
                return (
                    {
                        "prediction": "",
                        "status": "needs_review",
                        "failure": "projection_group_exceeds_128_components",
                        "failure_box": box,
                        "component_count": int(n),
                        "timing": {"line_total": time.perf_counter() - begin},
                        "numeric_extra_forwards": 0,
                        "cnn_forwards": {"guidance": 0, "final": 0},
                    },
                    {"unrecognized_line_mask": ink.copy()},
                    None,
                )
        _, parts = cc(ink)
        heights = [c["h"] for c in parts]
        if heights:
            q = float(np.percentile(heights, 90))
            selected = [h for h in heights if 0.5 * q <= h <= 2.2 * q]
            if not selected:
                context_H = float(np.median(heights))
                runtime_warnings.append(
                    "empty_height_selection_used_median_of_all_components"
                )
        times["preflight"] = time.perf_counter() - begin
        cropper = CurrentCropper(ViewAdapter(self))
        guidance = trace_guidance(cropper) if trace else None
        t = time.perf_counter()
        units, events, H = segment_line(ink, cropper, context_H=context_H)
        times["segmentation"] = time.perf_counter() - t
        if not units:
            return (
                {"prediction": "", "units": 0, "timing": times, "empty": True},
                None,
                None,
            )
        t = time.perf_counter()
        meta, data = pack(units, ink.shape, H)
        times["normalize_pack"] = time.perf_counter() - t
        t = time.perf_counter()
        raw, probs, views = self.classify(data["images"], keep_views=trace)
        times["recognition"] = time.perf_counter() - t
        if trace:
            data["actual_final_views"] = views
        t = time.perf_counter()
        decoded, nasal = apply(meta, data, raw, "combined")
        forced = []
        for i, u in enumerate(units):
            if u["forced"] is not None:
                forced.append(
                    {
                        "input_index": i,
                        "before": decoded[i]["label"],
                        "after": u["forced"],
                        "events": u["events"],
                    }
                )
                decoded[i]["label"] = u["forced"]
        pre_numeric = copy.deepcopy(decoded)
        options = self.digit_options(probs)
        decoded, numeric = decode(meta, decoded, options, "cautious")
        tokens = [p["label"] for p in decoded]
        joined, assembly_events = assembly.assemble_dandas(
            tokens, meta["lines"][0]["glyph_boxes"], list(range(len(tokens)))
        )
        text, encoding = assembly.encode_visual_parts("".join(joined))
        times["decode"] = time.perf_counter() - t
        result = {
            "seed": 17,
            "condition": "combined",
            "number_mode": "cautious",
            "prediction": text,
            "metadata": meta,
            "status": "recognized",
            "runtime_warnings": runtime_warnings,
            "raw_predictions": raw,
            "pre_numeric_predictions": pre_numeric,
            "predictions": decoded,
            "digit_options": options,
            "crop_events": events,
            "nasal_decisions": nasal,
            "punctuation_overrides": forced,
            "numeric_events": numeric,
            "assembly_events": assembly_events,
            "encoding_changes": encoding,
            "ink_conserved": True,
            "timing": times,
            "cnn_forwards": {k: self.forwards[k] - before[k] for k in before},
            "numeric_extra_forwards": 0,
            "score_note": "Top3 is the raw CNN distribution, not a probability for rule-imposed labels.",
        }
        gd = None
        if trace:
            records, images, masks, offsets, calls = guidance
            gd = {
                "metadata": {"predictions": records, "calls": calls, "views": 1},
                "arrays": {
                    "images": np.stack(images),
                    "mask_data": np.concatenate(masks),
                    "mask_offsets": np.array(offsets),
                },
            }
        times["line_total"] = time.perf_counter() - begin
        return result, data, gd
