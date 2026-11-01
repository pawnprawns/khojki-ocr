"""Run the full page/PDF OCR pipeline and write UTF-8 Khojki to stdout.

No reference text or manual line coordinates are accepted. PDF page numbers
are one-based; PDF rendering defaults to scale 2 (144 DPI). Use shell
redirection to save the text. Diagnostic files are optional.
"""

from pathlib import Path
import argparse, json, time, hashlib, sys
import cv2, numpy as np, pypdfium2 as pdfium
from .runtime import Engine, configure
from .layout import find_lines
from .preprocessing import foreground
from .encoding import validate


def save(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_page(gray, engine, trace=False):
    start = time.perf_counter()
    t = start
    ink, raw, preprocessing = foreground(gray, with_info=True)
    times = {"preprocess": time.perf_counter() - t}
    t = time.perf_counter()
    lines, audit, text, border = find_lines(ink, raw)
    times["line_layout"] = time.perf_counter() - t
    out = []
    arrays = []
    guidance = []
    t = time.perf_counter()
    for l in lines:
        pred, data, gd = engine.predict_line(l["ink"], trace)
        out.append(
            {
                "id": l["id"],
                "bbox": l["bbox"],
                "anchor_bbox": l["anchor_bbox"],
                "warnings": l["warnings"],
                **pred,
            }
        )
        arrays.append(data)
        guidance.append(gd)
    times["line_inference"] = time.perf_counter() - t
    times["page_compute"] = time.perf_counter() - start
    return (
        {
            "shape": list(gray.shape),
            "seed": 17,
            "lines": out,
            "text": "\n".join(l["prediction"] for l in out),
            "preprocessing": preprocessing,
            "status": (
                "layout_needs_review"
                if audit.get("layout_review_required")
                else (
                    "partial_needs_review"
                    if any(l.get("status") == "needs_review" for l in out)
                    else "processed_unreviewed"
                )
            ),
            "layout_audit": audit,
            "timing": times,
            "unicode_validation": validate("\n".join(l["prediction"] for l in out)),
            "model_loads_in_session": engine.model_loads,
            "numeric_extra_forwards": 0,
        },
        arrays,
        guidance,
        {
            "threshold_raw": raw,
            "foreground": ink,
            "text_foreground": text,
            "border_removed": border,
        },
    )


def save_page(dest, result, arrays, guidance, masks, gray):
    dest.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    for name, im in masks.items():
        cv2.imwrite(str(dest / f"{name}.png"), im)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for l, data, gd in zip(result["lines"], arrays, guidance):
        x, y, r, b = l["bbox"]
        cv2.rectangle(overlay, (x, y), (r - 1, b - 1), (40, 160, 30), 2)
        cv2.putText(
            overlay,
            l["id"],
            (x, max(15, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 20, 220),
            1,
        )
        if data is not None:
            np.savez_compressed(dest / f'{l["id"]}-inputs.npz', **data)
            if gd:
                np.savez_compressed(dest / f'{l["id"]}-guidance.npz', **gd["arrays"])
                save(dest / f'{l["id"]}-guidance.json', gd["metadata"])
    cv2.imwrite(str(dest / "line-overlay.png"), overlay)
    (dest / "text.txt").write_text(result["text"] + "\n", encoding="utf-8")
    result["timing"]["export_before_json"] = time.perf_counter() - start
    save(dest / "prediction.json", result)


def page_numbers(spec, count):
    if spec is None:
        return list(range(1, count + 1))
    values = []
    for token in spec.split(","):
        pair = token.split("-")
        a = int(pair[0])
        b = int(pair[-1])
        if len(pair) > 2 or not 1 <= a <= b <= count:
            raise ValueError("Invalid one-based page range")
        values.extend(range(a, b + 1))
    if len(set(values)) != len(values):
        raise ValueError("Repeated page number")
    return values


def main():
    start = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image", type=Path, help="Input page image (PNG, JPEG, TIFF, etc.)"
    )
    src.add_argument(
        "--pdf", type=Path, help="Input PDF; processes every page by default"
    )
    p.add_argument("--pages", help="PDF page list/ranges, e.g. 22-24")
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        help="Optional new directory for diagnostic images, JSON and per-page text",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="Also save model-input arrays in --output DIR",
    )
    a = p.parse_args()
    if a.output and a.output.exists():
        p.error("Refusing to overwrite an existing run")
    if a.trace and a.output is None:
        p.error("--trace requires --output DIR")
    if a.image and a.pages:
        p.error("--pages applies to --pdf")
    if not 0.5 <= a.scale <= 4:
        p.error("--scale must be between .5 and 4")
    source = a.pdf or a.image
    if not source.is_file():
        p.error(f"Input file does not exist: {source}")
    # Check input/ranges before allocating a model or creating run directories.
    doc = None
    try:
        doc = pdfium.PdfDocument(a.pdf) if a.pdf else None
        ids = page_numbers(a.pages, len(doc)) if doc is not None else [1]
        if a.image:
            preview = cv2.imread(str(a.image), 0)
            if preview is None:
                raise ValueError("Cannot read input image")
            if min(preview.shape) < 64:
                raise ValueError(
                    "Expected a page image at least 64 pixels in each dimension"
                )
    except (ValueError, pdfium.PdfiumError) as error:
        if doc is not None:
            doc.close()
        p.error(str(error))
    configure()
    engine = Engine()
    if a.output:
        a.output.mkdir(parents=True)
    source_sha = sha(source) if a.output else None
    # Preserve UTF-8 even when redirected on systems with another default encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    summary = []
    for page in ids:
        t = time.perf_counter()
        if doc:
            pg = doc[page - 1]
            bitmap = pg.render(scale=a.scale)
            rgb = np.array(bitmap.to_pil().convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            bitmap.close()
            pg.close()
        else:
            gray = cv2.imread(str(a.image), 0)
            if gray is None:
                raise ValueError("Cannot read input image")
        input_seconds = time.perf_counter() - t
        r, data, gd, masks = predict_page(gray, engine, a.trace)
        r["source"] = {
            "path": source.name,
            "sha256": source_sha,
            "page": page,
            "render_scale": a.scale if doc else None,
        }
        r["timing"]["input"] = input_seconds
        if a.output:
            save_page(a.output / f"page-{page:03}", r, data, gd, masks, gray)
        if summary:
            sys.stdout.write("\n")
        if r["text"]:
            sys.stdout.write(r["text"] + "\n")
        sys.stdout.flush()
        summary.append(
            {
                "page": page,
                "lines": len(r["lines"]),
                "status": r["status"],
                "regions_needing_review": sum(
                    l.get("status") == "needs_review" for l in r["lines"]
                ),
                "timing": r["timing"],
            }
        )
        if a.output:
            print(json.dumps(summary[-1]), file=sys.stderr, flush=True)
    if doc:
        doc.close()
    partial = any(
        p["regions_needing_review"] or p["status"] == "layout_needs_review"
        for p in summary
    )
    if a.output:
        save(
            a.output / "run.json",
            {
                "source": source.name,
                "source_sha256": source_sha,
                "pages": summary,
                "model_loads": engine.model_loads,
                "model_setup_seconds": engine.init_seconds,
                "cnn_forwards": engine.forwards,
                "numeric_extra_forwards": 0,
                "total_seconds_after_imports": time.perf_counter() - start,
                "status": "partial_needs_review" if partial else "processed_unreviewed",
                "scope": "Automatic line proposals; all output unscored until a separate evaluation. Printed Khojki only validated; handwriting unsupported by existing evidence.",
            },
        )
    if partial:
        print(
            "Some pages or regions need review; any recognized text has been written.",
            file=sys.stderr,
        )
    return 2 if partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
