"""Post-inference evaluation. Reference labels never enter the recognizer."""

from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import unicodedata

from .encoding import validate
from .metrics import character_errors, iou
from .model import ASSETS


def evaluate(prediction, references, minimum_iou=0.5):
    """One-to-one geometric matching on the annotated subset of a page.

    Unannotated page regions are not counted as false positives. A missing
    reference line is scored against an empty string, producing deletions.
    Text is not consulted during matching.
    """
    candidates = sorted(
        (
            (iou(ref["bbox"], line["bbox"]), i, j)
            for i, ref in enumerate(references["lines"])
            for j, line in enumerate(prediction["lines"])
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    matched, used = {}, set()
    for overlap, i, j in candidates:
        if overlap >= minimum_iou and i not in matched and j not in used:
            matched[i] = (j, overlap)
            used.add(j)
    rows = []
    for i, ref in enumerate(references["lines"]):
        match = matched.get(i)
        line = prediction["lines"][match[0]] if match else None
        text = line["prediction"] if line else ""
        rows.append(
            {
                "id": ref["id"],
                "original_pdf_page": ref["page"],
                "reference_box": ref["bbox"],
                "automatic_line": line["id"] if line else None,
                "automatic_box": line["bbox"] if line else None,
                "matching_iou": match[1] if match else 0.0,
                "reference": ref["text"],
                "prediction": text,
                "metric": character_errors(ref["text"], text),
                "unicode_validation": validate(text),
            }
        )
    totals = {
        k: sum(r["metric"][k] for r in rows)
        for k in [
            "errors",
            "reference_characters",
            "substitutions",
            "deletions",
            "insertions",
        ]
    }
    if not totals["reference_characters"]:
        raise ValueError("Cannot score an empty reference set")
    totals["cer"] = totals["errors"] / totals["reference_characters"]
    return {
        "scope": references["split"],
        "reference_status": references["status"],
        "method": "NFC code-point Levenshtein; punctuation and marks retained; lines scored separately",
        "matching": "one-to-one bounding-box IoU; no text matching",
        "minimum_iou": minimum_iou,
        "summary": totals,
        "line_detection": {
            "annotated": len(rows),
            "matched": len(matched),
            "missed": len(rows) - len(matched),
            "false_positives_full_page": None,
            "reason": f"Only {len(rows)} target lines are annotated, not the entire page",
        },
        "lines": rows,
    }


def verify_source(prediction, references):
    source = prediction["source"]
    if source.get("render_scale") is not None:
        if (
            source["sha256"] != references["source_pdf_sha256"]
            or source["render_scale"] != references["render_scale"]
            or {r["page"] for r in references["lines"]} != {source["page"]}
        ):
            raise ValueError(
                "Reference PDF hash, page or rendering scale does not match predictions"
            )
    elif {r["image_sha256"] for r in references["lines"]} != {source["sha256"]}:
        raise ValueError("Reference image hash does not match predictions")


def describe(text):
    if not text:
        return "(none)"
    return " + ".join(
        f'U+{ord(c):04X} {unicodedata.name(c, "UNASSIGNED")}' for c in text
    )


def write_review(result, references, reference_path, output):
    """Portable static review, with copied excerpts and the licensed font."""
    assets = output / "assets"
    assets.mkdir()
    for name in ["NotoSansKhojki-Regular.ttf", "OFL.txt"]:
        shutil.copy2(ASSETS / name, assets / name)
    cards = []
    esc = html.escape
    for row, ref in zip(result["lines"], references["lines"]):
        image = reference_path.resolve().parent.parent / ref["crop"]
        name = f'{row["id"]}.png'
        shutil.copy2(image, assets / name)
        edits = "".join(
            f'<tr><td>{esc(a["op"])}</td><td>{esc(describe(a["reference"]))}</td>'
            f'<td>{esc(describe(a["hypothesis"]))}</td></tr>'
            for a in row["metric"]["alignment"]
            if a["op"] != "equal"
        )
        m = row["metric"]
        cards.append(
            f"""<article id="{esc(row['id'])}">
          <h2>{esc(row['id'])} · {m['errors']}/{m['reference_characters']} edits · {100*m['cer']:.2f}% CER</h2>
          <img class="line" src="assets/{name}" alt="Source excerpt {esc(row['id'])}">
          <p class="caption">Reviewed reference</p><div class="kh">{esc(row['reference'])}</div>
          <p class="caption">Automatic prediction</p><div class="kh">{esc(row['prediction'])}</div>
          <p>Matched {esc(row['automatic_line'] or 'no automatic region')} by image geometry; IoU {row['matching_iou']:.3f}.</p>
          <table><tr><th>Edit</th><th>Reference</th><th>Prediction</th></tr>{edits}</table></article>"""
        )
    m = result["summary"]
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Khojki OCR: line evaluation</title>
    <style>@font-face{{font-family:Khojki;src:url('assets/NotoSansKhojki-Regular.ttf')}}
    body{{font:16px/1.55 system-ui,sans-serif;background:#f3f1eb;color:#21372f;margin:0}}
    main{{max-width:1100px;margin:auto;padding:36px 24px}}h1{{font-size:34px}}h2{{font-size:22px}}
    article{{background:white;border:1px solid #cbd8ca;border-radius:10px;padding:24px;margin:28px 0}}
    .kh{{font:27px/1.8 Khojki,serif;overflow-wrap:anywhere}}.line{{width:100%;height:auto}}
    .caption{{font-size:13px;font-weight:bold;margin-bottom:2px;color:#536850}}
    .note{{border-left:4px solid #c58b2b;padding:16px;background:#fff2d6}}
    table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #d9dfd6}}
    </style></head><body><main><h1>Khojki OCR evaluation</h1>
    <p><b>{m['errors']} / {m['reference_characters']} = {100*m['cer']:.2f}% CER</b> · {m['substitutions']} substitutions, {m['deletions']} deletion(s), {m['insertions']} insertion(s).</p>
    <div class="note">These are exposed development lines. References were chart-assisted and visually reviewed, not expert-certified. This score does not establish performance on unseen books or handwriting. References are used only after full-page inference.</div>
    <p>The full-page detector matched {result['line_detection']['matched']}/{result['line_detection']['annotated']} annotated targets. Other page regions are unscored. Marks and punctuation count in CER. <a href="evaluation.json">Machine-readable scores and complete alignments</a>.</p>
    {''.join(cards)}</main></body></html>"""
    (output / "review.html").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing evaluation")
    prediction = json.loads(args.predictions.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))
    verify_source(prediction, references)
    result = evaluate(prediction, references)
    result["prediction_sha256"] = hashlib.sha256(
        args.predictions.read_bytes()
    ).hexdigest()
    result["reference_sha256"] = hashlib.sha256(
        args.references.read_bytes()
    ).hexdigest()
    args.output.mkdir(parents=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_review(result, references, args.references, args.output)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
