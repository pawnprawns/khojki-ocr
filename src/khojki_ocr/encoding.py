"""Encoding checks are not a substitute for visual transcription checks."""

import unicodedata

# Unicode 16, Table 15-4. Longest sequences must win: A+AA+AI is AU,
# not independent AA followed by a second vowel sign. The last two entries
# accommodate the left-to-right visual-component order produced by segmentation.
ATOMIC_RULES = {
    "\U00011200\U0001122c\U00011231": "\U00011207",
    "\U00011240\U0001122e": "\U00011202",
    "\U00011206\U0001122c": "\U00011203",
    "\U00011200\U00011231": "\U00011205",
    "\U00011200\U00011233": "\U00011207",
    "\U00011200\U0001122c": "\U00011201",
    "\U0001122c\U00011230": "\U00011232",
    "\U0001122c\U00011231": "\U00011233",
    "\U00011230\U0001122c": "\U00011232",
    "\U00011231\U0001122c": "\U00011233",
}


def structural_warnings(text):
    """Conservative review flags, never a language correctness certificate."""
    warnings = []
    base_seen = vowel_seen = False
    for i, ch in enumerate(text):
        cp = ord(ch)
        if (0x11200 <= cp <= 0x1122B and cp != 0x11212) or cp in (0x1123F, 0x11240):
            base_seen, vowel_seen = True, False
        elif 0x1122C <= cp <= 0x11233:
            if not base_seen:
                warnings.append(
                    {"index": i, "kind": "vowel_sign_without_preceding_base"}
                )
            elif vowel_seen:
                warnings.append(
                    {"index": i, "kind": "multiple_vowel_signs_require_review"}
                )
            vowel_seen = True
        elif cp in range(0x11234, 0x11238):
            continue
        else:
            base_seen = vowel_seen = False
    return warnings


def validate(text):
    invalid = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        allowed = (
            ch in " \n\t-"
            or cp == 0xFFFD
            or 0x0AE6 <= cp <= 0x0AEF
            or (0x11200 <= cp <= 0x11241 and cp != 0x11212)
        )
        if not allowed:
            invalid.append(
                {
                    "index": i,
                    "codepoint": f"U+{cp:04X}",
                    "name": unicodedata.name(ch, "UNASSIGNED"),
                }
            )
    return {
        "utf8_roundtrip": text.encode("utf-8").decode("utf-8") == text,
        "invalid": invalid,
        "unknown_glyphs": text.count("\ufffd"),
        "codepoints": [f"U+{ord(ch):04X}" for ch in text],
        "structural_warnings": structural_warnings(text),
    }


def normalize(text):
    # No punctuation or whitespace deletion: those remain measurable errors.
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n"))


def _mapped_parts(text):
    i = 0
    while i < len(text):
        source = next((s for s in ATOMIC_RULES if text.startswith(s, i)), text[i])
        yield i, source, ATOMIC_RULES.get(source, source)
        i += len(source)


def encode_visual_parts(text):
    """Fold chart-defined visual parts; NFC alone does not encode them atomically."""
    output, changes = [], []
    for _, source, target in _mapped_parts(text):
        output.append(target)
        if source != target:
            changes.append(
                {
                    "from": [f"U+{ord(c):04X}" for c in source],
                    "to": f"U+{ord(target):04X}",
                    "occurrences": 1,
                }
            )
    return normalize("".join(output)), changes


def encode_with_boxes(text, boxes):
    """Keep code-point-to-image provenance when an atomic mapping shortens text."""
    if len(text) != len(boxes):
        raise ValueError("Every raw code point must have a source box")
    out = []
    for i, source, target in _mapped_parts(text):
        origins = boxes[i : i + len(source)]
        merged = [
            min(b[0] for b in origins),
            min(b[1] for b in origins),
            max(b[2] for b in origins),
            max(b[3] for b in origins),
        ]
        out.append((target, merged))
    return "".join(x[0] for x in out), [x[1] for x in out]
