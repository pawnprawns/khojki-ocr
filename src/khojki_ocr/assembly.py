"""Adjacent danda assembly and Unicode visual-part folding."""

from .component_grouping import xgap
from .encoding import encode_visual_parts


def assemble_dandas(tokens, boxes, ids):
    after = tokens[:]
    events = []
    i = 0
    while i + 1 < len(boxes):
        a, b = boxes[i], boxes[i + 1]
        h1 = a[3] - a[1]
        h2 = b[3] - b[1]
        h = (h1 + h2) / 2
        if (
            after[i] == after[i + 1] == "𑈸"
            and xgap(a, b) <= 0.2 * h
            and abs(a[1] - b[1]) <= 0.15 * h
            and abs(a[3] - b[3]) <= 0.15 * h
        ):
            after[i] = "𑈹"
            after[i + 1] = ""
            events.append(
                {
                    "kind": "double_danda_assembly",
                    "input_indices": [ids[i], ids[i + 1]],
                    "boxes": [a, b],
                }
            )
            i += 2
        else:
            i += 1
    return after, events
