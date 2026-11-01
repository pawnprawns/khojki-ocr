"""Lossless storage of actual guidance and recognition inputs."""

import copy, hashlib
import numpy as np
from .glyphs import normalize_glyph


def trace_guidance(cropper):
    original = cropper.classify
    records = []
    images = []
    masks = []
    offsets = [0]
    keys = {}
    calls = []

    def wrapped(units):
        result, actual = original(units)
        ids = []
        for u, im, p in zip(units, actual, result):
            key = hashlib.sha256(im.tobytes()).hexdigest()
            assert np.array_equal(
                im, np.uint8(np.rint(normalize_glyph(u["ink"]) * 255))
            )
            if key not in keys:
                keys[key] = len(records)
                images.append(im)
                masks.append(u["ink"].ravel())
                offsets.append(offsets[-1] + u["ink"].size)
                records.append(
                    {
                        "input_index": keys[key],
                        "shape": list(u["ink"].shape),
                        "prediction": copy.deepcopy(p),
                        "sha256": key,
                    }
                )
            ids.append(keys[key])
        calls.append(ids)
        return result, actual

    cropper.classify = wrapped
    return records, images, masks, offsets, calls


def pack(units, shape, H):
    images = []
    masks = []
    offsets = [0]
    rows = []
    for i, u in enumerate(units):
        images.append(np.uint8(np.rint(normalize_glyph(u["ink"]) * 255)))
        masks.append(u["ink"].ravel())
        offsets.append(offsets[-1] + u["ink"].size)
        rows.append(
            {k: v for k, v in u.items() if k != "ink"}
            | {"input_index": i, "line_id": "line", "page": 0, "id": f"unit-{i:03}"}
        )
    meta = {
        "units": rows,
        "lines": [
            {
                "id": "line",
                "page": 0,
                "bbox": [0, 0, shape[1], shape[0]],
                "input_indices": list(range(len(units))),
                "glyph_boxes": [u["box"] for u in units],
                "geometry_H": H,
            }
        ],
    }
    data = {
        "images": np.stack(images),
        "mask_data": np.concatenate(masks),
        "mask_offsets": np.array(offsets),
    }
    return meta, data
