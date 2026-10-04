"""Scored eval for the flat-lay pipeline: detect -> crop -> tag, run against a
hand-labeled set of photos, so we get actual numbers instead of eyeballing.

Ground truth lives in eval/labels.json, photos in eval/photos/. Each photo lists
the items a person sees in it and the category each one should get. "category"
can be a list when more than one answer is fair (e.g. a dress has no exact
category in our taxonomy, so it's labeled ["top", "bottom"]). Items marked
"ignore": true (underwear, swimwear) are neither required nor penalized: missing
them costs nothing, and a detection that lands on one isn't counted as extra.
Non-clothing (books, towels, perfume) is simply left unlabeled, so detecting it
counts against precision.

Predicted items are matched to labeled items by category (best one-to-one
matching), so we don't need hand-drawn boxes. That means a correct count with
the wrong categories scores low, but two swapped items of the same category
can't be told apart — fine for what we're measuring.

Usage:
    python eval_flatlay.py                 # detect + tag every labeled photo
    python eval_flatlay.py --detect-only   # just item counts, no tagging calls
    python eval_flatlay.py --only laid_out_test.jpg

Each run is saved to eval/results/<timestamp>.json so numbers can be compared
across prompt/model changes.
"""

import argparse
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.detection import crop_detections, detect_garments
from app.ingestion import tag_photo

CATEGORIES = ["top", "bottom", "shoes", "outerwear", "accessory"]

EVAL_DIR = Path(__file__).parent / "eval"
LABELS_PATH = EVAL_DIR / "labels.json"
PHOTOS_DIR = EVAL_DIR / "photos"
RESULTS_DIR = EVAL_DIR / "results"


def accepted(label_item: dict) -> set[str]:
    # Ignored items may omit a category, in which case any prediction can land on them.
    cat = label_item.get("category", CATEGORIES)
    return set(cat) if isinstance(cat, list) else {cat}


def match(predicted: list[str], labeled: list[dict]) -> list[tuple[int, int]]:
    """Maximum one-to-one matching of predicted categories to labeled items
    (simple augmenting-path bipartite matching — the sets here are tiny).
    Returns (pred_index, label_index) pairs."""
    label_owner: dict[int, int] = {}

    def try_assign(p: int, seen: set[int]) -> bool:
        for li, item in enumerate(labeled):
            if predicted[p] in accepted(item) and li not in seen:
                seen.add(li)
                if li not in label_owner or try_assign(label_owner[li], seen):
                    label_owner[li] = p
                    return True
        return False

    for p in range(len(predicted)):
        try_assign(p, set())
    return [(p, li) for li, p in label_owner.items()]


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def evaluate_photo(entry: dict, detect_only: bool, crops_dir: Path) -> dict:
    photo_path = PHOTOS_DIR / entry["file"]
    labeled = [it for it in entry["items"] if not it.get("ignore")]
    ignored = [it for it in entry["items"] if it.get("ignore")]

    detections = detect_garments(photo_path)
    result = {
        "file": entry["file"],
        "labeled_count": len(labeled),
        "ignored_count": len(ignored),
        "detected_count": len(detections.items),
        "detected_labels": [d.label for d in detections.items],
    }
    if detect_only:
        return result

    crop_paths = crop_detections(photo_path, detections, output_dir=crops_dir)

    # Mirror /items/batch/detect: a crop that fails to tag is skipped, not fatal.
    kept_dets, predicted, tag_failures = [], [], []
    for det, crop_path in zip(detections.items, crop_paths):
        try:
            predicted.append(tag_photo(crop_path, hint=det.label).category)
            kept_dets.append(det)
        except Exception as exc:
            tag_failures.append(f"{det.label}: {type(exc).__name__}")
    detections.items = kept_dets
    result["tag_failures"] = tag_failures

    pairs = match(predicted, labeled)
    matched_preds = {p for p, _ in pairs}
    matched_labels = {li for _, li in pairs}

    # Leftover predictions that land on an ignored item are dropped from scoring.
    leftover = [p for p in range(len(predicted)) if p not in matched_preds]
    ignored_pairs = match([predicted[p] for p in leftover], ignored)
    ignored_preds = {leftover[i] for i, _ in ignored_pairs}

    result.update({
        "predicted_categories": predicted,
        "scored_detected_count": len(predicted) - len(ignored_preds),
        "matched": len(pairs),
        "missed": [labeled[li]["name"] for li in range(len(labeled)) if li not in matched_labels],
        "extra": [
            f"{detections.items[p].label} -> {predicted[p]}"
            for p in leftover if p not in ignored_preds
        ],
        "per_category": {
            # Per-category recall only counts single-category labels, so a
            # dress doesn't get split across two buckets.
            c: {
                "labeled": sum(1 for it in labeled if accepted(it) == {c}),
                "matched": sum(1 for _, li in pairs if accepted(labeled[li]) == {c}),
            }
            for c in CATEGORIES
        },
    })
    return result


def count_error(r: dict) -> int:
    """How far the detected count falls outside the acceptable range: every
    labeled item, plus optionally any of the ignored ones."""
    low, high = r["labeled_count"], r["labeled_count"] + r["ignored_count"]
    return max(low - r["detected_count"], r["detected_count"] - high, 0)


def summarize(results: list[dict], detect_only: bool) -> dict:
    total_labeled = sum(r["labeled_count"] for r in results)
    summary = {
        "photos": len(results),
        "labeled_items": total_labeled,
        "ignored_items": sum(r["ignored_count"] for r in results),
        "detected_items": sum(r["detected_count"] for r in results),
        "exact_count_rate": ratio(sum(count_error(r) == 0 for r in results), len(results)),
        "mean_abs_count_error": ratio(sum(count_error(r) for r in results), len(results)),
    }
    if detect_only:
        return summary

    matched = sum(r["matched"] for r in results)
    precision = ratio(matched, sum(r["scored_detected_count"] for r in results))
    recall = ratio(matched, total_labeled)
    per_cat = {}
    for c in CATEGORIES:
        lab = sum(r["per_category"][c]["labeled"] for r in results)
        hit = sum(r["per_category"][c]["matched"] for r in results)
        if lab:
            per_cat[c] = {"labeled": lab, "matched": hit, "recall": ratio(hit, lab)}
    summary.update({
        "matched_items": matched,
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0,
        "per_category_recall": per_cat,
    })
    return summary


def print_report(results: list[dict], summary: dict, detect_only: bool) -> None:
    print("\nPer photo:")
    for r in results:
        line = f"  {r['file']:<32} labeled={r['labeled_count']:<3} detected={r['detected_count']:<3}"
        if not detect_only:
            line += f" matched={r['matched']}"
        print(line)
        if not detect_only:
            for name in r["missed"]:
                print(f"      missed: {name}")
            for extra in r["extra"]:
                print(f"      extra:  {extra}")
            for failure in r["tag_failures"]:
                print(f"      tag failed (skipped): {failure}")

    print("\nSummary:")
    print(f"  photos:               {summary['photos']}")
    print(f"  labeled / detected:   {summary['labeled_items']} / {summary['detected_items']}"
          f"  ({summary['ignored_items']} ignored items not scored)")
    print(f"  exact item count:     {summary['exact_count_rate']:.0%} of photos")
    print(f"  mean |count error|:   {summary['mean_abs_count_error']:.2f} items per photo")
    if not detect_only:
        print(f"  precision:            {summary['precision']:.0%}  (detected items that match a real item + category)")
        print(f"  recall:               {summary['recall']:.0%}  (real items found with the right category)")
        print(f"  F1:                   {summary['f1']:.0%}")
        for c, s in summary["per_category_recall"].items():
            print(f"    {c:<10} recall {s['recall']:.0%}  ({s['matched']}/{s['labeled']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detect-only", action="store_true", help="skip tagging; score item counts only")
    parser.add_argument("--only", help="run a single photo by filename")
    args = parser.parse_args()

    entries = json.loads(LABELS_PATH.read_text())["photos"]
    if args.only:
        entries = [e for e in entries if e["file"] == args.only]
        if not entries:
            print(f"No labeled photo named {args.only!r} in {LABELS_PATH}")
            sys.exit(1)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, entry in enumerate(entries, 1):
            print(f"[{i}/{len(entries)}] {entry['file']}...")
            results.append(evaluate_photo(entry, args.detect_only, Path(tmp) / str(i)))

    summary = summarize(results, args.detect_only)
    print_report(results, summary, args.detect_only)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "detect-only" if args.detect_only else "detect+tag",
        "summary": summary,
        "photos": results,
    }, indent=2))
    print(f"\nSaved to {out_path}")
