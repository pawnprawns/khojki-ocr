"""Frozen native-scale nasal completion, dot ownership and evidence veto."""

import copy
import cv2
import numpy as np

AN = "\U00011234"
AA = "\U0001122c"


def round_component(box, area, H):
    x, y, w, h = box
    return (
        0.12 * H <= min(w, h)
        and max(w, h) <= 0.50 * H
        and 0.65 <= w / h <= 1.55
        and area >= 0.025 * H * H
        and 0.30 <= area / (w * h) <= 0.96
    )


def components(ink):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    return lab, [
        dict(k=k, box=stats[k, :4].tolist(), area=int(stats[k, 4])) for k in range(1, n)
    ]


def attached_evidence(ink, label, H):
    lab, comps = components(ink)
    hits = []
    for c in comps:
        x, y, w, h = c["box"]
        if not round_component(c["box"], c["area"], H):
            continue
        bodies = [
            b
            for b in comps
            if b["k"] != c["k"] and b["box"][3] >= 0.5 * H and b["area"] >= 0.08 * H * H
        ]
        if not bodies:
            continue
        top = min(b["box"][1] for b in bodies)
        left = min(b["box"][0] for b in bodies)
        right = max(b["box"][0] + b["box"][2] for b in bodies)
        if (
            y + h / 2 <= top + 0.15 * H
            and y + h <= top + 0.30 * H
            and left - 0.25 * H <= x + w / 2 <= right + 0.25 * H
        ):
            hits.append({"kind": "detached_upper_round_mark", "box": [x, y, w, h]})
    # Some printed rings touch the AA stem. A hole at the top of a thin AA
    # fragment is evidence even though connected-components cannot detach it.
    # This condition deliberately requires the CNN to identify the AA stroke.
    if (
        label.replace(AN, "") == AA
        and ink.shape[1] <= 0.50 * H
        and ink.shape[0] >= 0.8 * H
    ):
        contours, hierarchy = cv2.findContours(
            ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for k, contour in enumerate(contours):
            if hierarchy[0, k, 3] < 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) >= 0.08 * H and 0.60 <= w / h <= 1.55 and y + h <= 0.45 * H:
                hits.append({"kind": "connected_AA_upper_ring", "box": [x, y, w, h]})
    return hits


def isolated_evidence(ink, box, prev_ink, prev_box, H):
    if prev_ink is None:
        return []
    lab, comps = components(ink)
    if len(comps) != 1 or not round_component(comps[0]["box"], comps[0]["area"], H):
        return []
    if prev_ink.shape[0] < 0.50 * H or prev_ink.shape[1] < 0.35 * H:
        return []
    _, owner_components = components(prev_ink)
    bodies = [
        c
        for c in owner_components
        if c["box"][3] >= 0.50 * H and c["box"][2] >= 0.35 * H
    ]
    if not bodies:
        return []
    # A detached E stroke can occupy the owner's top 15% of ink. Use the
    # dominant substantial component, rather than mixing vowel and body pixels.
    body = max(bodies, key=lambda c: c["area"])
    upper_body = float(body["box"][1] + prev_box[1])
    # An upper-right mark follows its owner in the existing left-to-right units.
    if (
        0 <= box[0] - prev_box[2] <= 0.40 * H
        and (box[1] + box[3]) / 2 <= upper_body - 0.05 * H
        and box[3] <= upper_body + 0.15 * H
    ):
        return [
            {
                "kind": "isolated_upper_right_dot",
                "box": [0, 0, ink.shape[1], ink.shape[0]],
                "owner_gap": box[0] - prev_box[2],
                "owner_upper_body": upper_body,
            }
        ]
    return []


def get_masks(meta, data):
    return [
        data["mask_data"][
            data["mask_offsets"][i] : data["mask_offsets"][i + 1]
        ].reshape(u["box"][3] - u["box"][1], u["box"][2] - u["box"][0])
        for i, u in enumerate(meta["units"])
    ]


def apply(meta, data, predictions, condition):
    """Ablations alter only output nasal decisions; all image inputs stay fixed."""
    output = copy.deepcopy(predictions)
    masks = get_masks(meta, data)
    decisions = []
    for line in meta["lines"]:
        ids = line["input_indices"]
        H = float(np.median([masks[i].shape[0] for i in ids]))
        for j, i in enumerate(ids):
            before = predictions[i]["label"]
            previous = ids[j - 1] if j else None
            attached = attached_evidence(masks[i], before, H)
            isolated = isolated_evidence(
                masks[i],
                meta["units"][i]["box"],
                masks[previous] if previous is not None else None,
                meta["units"][previous]["box"] if previous is not None else None,
                H,
            )
            after = before
            action = None
            if (
                condition in ["attached_completion", "combined"]
                and attached
                and AN not in after
            ):
                # A base/vowel or a dependent vowel fragment can precede a nasal.
                # Exclude digits, punctuation, and standalone signs.
                if after and (
                    0x11200 <= ord(after[0]) <= 0x1122B
                    or ord(after[0]) == 0x1123F
                    or after[0] == AA
                ):
                    after += AN
                    action = "append_nasal_from_attached_evidence"
            if condition in ["isolated_dot", "combined"] and isolated:
                after = AN
                action = "decode_dot_at_native_scale"
            if (
                condition in ["evidence_veto", "combined"]
                and AN in after
                and not (attached or isolated)
            ):
                if after != AN:
                    after = after.replace(AN, "")
                    action = "remove_unsupported_nasal"
                else:
                    _, cc = components(masks[i])
                    if len(cc) == 1 and round_component(cc[0]["box"], cc[0]["area"], H):
                        decisions.append(
                            {
                                "input_index": i,
                                "line_id": line["id"],
                                "H": H,
                                "before": before,
                                "after": after,
                                "attached": attached,
                                "isolated": isolated,
                                "owner_index": None,
                                "action": "abstain_round_dot_owner_uncertain",
                            }
                        )
                        continue
                    alternatives = [
                        p["label"]
                        for p in predictions[i]["top3"]
                        if AN not in p["label"]
                    ]
                    if alternatives:
                        after = alternatives[0]
                        action = "use_best_non_nasal_top3_alternative"
                    else:
                        action = "abstain_no_non_nasal_alternative"
            output[i]["label"] = after
            if attached or isolated or AN in before or before != after:
                decisions.append(
                    {
                        "input_index": i,
                        "line_id": line["id"],
                        "H": H,
                        "before": before,
                        "after": after,
                        "attached": attached,
                        "isolated": isolated,
                        "owner_index": previous if isolated else None,
                        "action": action,
                    }
                )
    return output, decisions
