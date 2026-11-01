"""Code-point Levenshtein alignment, including punctuation and spaces."""

from .encoding import normalize


def character_errors(reference, hypothesis):
    a, b = normalize(reference), normalize(hypothesis)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    i, j = len(a), len(b)
    ops = []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(
                {
                    "op": "equal" if a[i - 1] == b[j - 1] else "substitute",
                    "ref_index": i - 1,
                    "hyp_index": j - 1,
                    "reference": a[i - 1],
                    "hypothesis": b[j - 1],
                }
            )
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(
                {
                    "op": "delete",
                    "ref_index": i - 1,
                    "hyp_index": None,
                    "reference": a[i - 1],
                    "hypothesis": "",
                }
            )
            i -= 1
        else:
            ops.append(
                {
                    "op": "insert",
                    "ref_index": None,
                    "hyp_index": j - 1,
                    "reference": "",
                    "hypothesis": b[j - 1],
                }
            )
            j -= 1
    ops.reverse()
    s = sum(x["op"] == "substitute" for x in ops)
    d = sum(x["op"] == "delete" for x in ops)
    ins = sum(x["op"] == "insert" for x in ops)
    return {
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "errors": s + d + ins,
        "reference_characters": len(a),
        "cer": (s + d + ins) / len(a) if a else (0.0 if not b else None),
        "alignment": ops,
    }


def iou(a, b):
    l, t = max(a[0], b[0]), max(a[1], b[1])
    r, bot = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, r - l) * max(0, bot - t)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0
