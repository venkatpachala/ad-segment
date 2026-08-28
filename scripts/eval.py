#!/usr/bin/env python3
"""Segment-level IoU eval against data/ground_truth.json.

Usage:
    # Single prediction evaluation:
    python scripts/eval.py data/predictions/pred_ujFWRFYLGjY.json ujFWRFYLGjY

    # Batch evaluation of all predictions in a directory:
    python scripts/eval.py data/runs/ --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.temporal import segment_iou  # noqa: E402


def video_id_from_url(url: str) -> str:
    u = (url or "").strip().split("?")[0].rstrip("/")
    if "watch" in u and "v=" in (url or ""):
        for part in (url or "").split("?", 1)[-1].split("&"):
            if part.startswith("v="):
                return part[2:]
    return u.split("/")[-1]


def load_gt() -> dict:
    gt_file = ROOT / "data" / "ground_truth.json"
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    return json.loads(gt_file.read_text(encoding="utf-8"))


def evaluate_single(
    pred_doc: dict,
    gold_segments: list[dict],
    iou_thresh: float = 0.5,
) -> dict:
    pred_segs = pred_doc.get("segments") or []
    used_p = set()
    tp = 0
    ious = []
    matches = []

    for g in gold_segments:
        best_i, best_iou = -1, 0.0
        for i, p in enumerate(pred_segs):
            if i in used_p:
                continue
            iou = segment_iou(g["start_s"], g["end_s"], p["start_s"], p["end_s"])
            if iou > best_iou:
                best_iou, best_i = iou, i

        if best_iou >= iou_thresh and best_i >= 0:
            used_p.add(best_i)
            tp += 1
            ious.append(best_iou)
            p = pred_segs[best_i]
            matches.append({
                "gold_id": g["id"],
                "iou": round(best_iou, 3),
                "status": "TP",
                "start_err_s": round(abs(p["start_s"] - g["start_s"]), 3),
                "end_err_s": round(abs(p["end_s"] - g["end_s"]), 3),
            })
        else:
            matches.append({"gold_id": g["id"], "iou": round(best_iou, 3), "status": "FN"})

    fp = len(pred_segs) - len(used_p)
    fn = len(gold_segments) - tp

    prec = tp / (tp + fp) if tp + fp else (1.0 if not gold_segments and not pred_segs else 0.0)
    rec = tp / (tp + fn) if tp + fn else (1.0 if not gold_segments and not pred_segs else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou = sum(ious) / len(ious) if ious else (1.0 if not gold_segments and not pred_segs else 0.0)
    tp_matches = [m for m in matches if m["status"] == "TP"]
    mean_start_err = (
        sum(m["start_err_s"] for m in tp_matches) / len(tp_matches) if tp_matches else None
    )
    mean_end_err = (
        sum(m["end_err_s"] for m in tp_matches) / len(tp_matches) if tp_matches else None
    )

    return {
        "gold_count": len(gold_segments),
        "pred_count": len(pred_segs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "mean_iou": round(mean_iou, 3),
        "mean_start_err_s": None if mean_start_err is None else round(mean_start_err, 3),
        "mean_end_err_s": None if mean_end_err is None else round(mean_end_err, 3),
        "matches": matches,
    }


def match_video(pred_path: Path, url_substr: str, iou_thresh: float = 0.5) -> dict:
    gt_doc = load_gt()
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    matches = [v for v in gt_doc["videos"] if url_substr in v["source"]["url"]]
    if not matches:
        raise ValueError(f"No video matching '{url_substr}' in ground_truth.json")
    entry = matches[0]
    gold = entry.get("segments") or []
    res = evaluate_single(pred, gold, iou_thresh)

    print(f"\nEvaluation for: {entry['source']['url']}")
    print(f"Gold segments: {res['gold_count']} | Predicted segments: {res['pred_count']}")
    print(f"TP: {res['tp']} | FP: {res['fp']} | FN: {res['fn']}")
    print(f"Precision: {res['precision']:.3f} | Recall: {res['recall']:.3f} | F1: {res['f1']:.3f} | Mean IoU: {res['mean_iou']:.3f}")
    if res.get("mean_start_err_s") is not None:
        print(f"Boundary |start|: {res['mean_start_err_s']:.2f}s | |end|: {res['mean_end_err_s']:.2f}s")
    for m in res["matches"]:
        extra = ""
        if m["status"] == "TP":
            extra = f" start_err={m['start_err_s']}s end_err={m['end_err_s']}s"
        print(f"  Segment {m['gold_id']}: {m['status']} (IoU={m['iou']}){extra}")
    return res


def evaluate_all(predictions_dir: Path, iou_thresh: float = 0.5) -> None:
    gt_doc = load_gt()
    print(f"\n========================================================")
    print(f"  BENCHMARK EVALUATION (IoU Threshold >= {iou_thresh})")
    print(f"========================================================")

    total_gold, total_pred, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0
    all_ious = []
    rows = []

    for v in gt_doc["videos"]:
        url = v["source"]["url"]
        gold = v.get("segments") or []
        vid_id = video_id_from_url(url)
        if not vid_id:
            rows.append({"url": url, "kind": v["source"].get("kind", "-"), "status": "MISSING_PRED", "f1": 0.0, "prec": 0.0, "rec": 0.0})
            total_gold += len(gold)
            total_fn += len(gold)
            continue

        # Look for matching prediction json file
        json_files = list(predictions_dir.glob("*.json")) + list(predictions_dir.rglob("*.json"))
        candidates = [f for f in json_files if vid_id in f.name]
        if not candidates:
            candidates = [
                f for f in json_files
                if vid_id in f.read_text(encoding="utf-8", errors="ignore")
            ]
        if not candidates:
            rows.append({"url": vid_id, "kind": v["source"].get("kind", "-"), "status": "MISSING_PRED", "f1": 0.0, "prec": 0.0, "rec": 0.0})
            total_gold += len(gold)
            total_fn += len(gold)
            continue

        pred_file = candidates[0]
        pred_doc = json.loads(pred_file.read_text(encoding="utf-8"))
        res = evaluate_single(pred_doc, gold, iou_thresh)

        total_gold += res["gold_count"]
        total_pred += res["pred_count"]
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]
        for m in res["matches"]:
            if m["status"] == "TP":
                all_ious.append(m["iou"])

        rows.append({
            "url": vid_id,
            "kind": v["source"].get("kind", "-"),
            "gold": res["gold_count"],
            "pred": res["pred_count"],
            "tp": res["tp"],
            "fp": res["fp"],
            "fn": res["fn"],
            "prec": res["precision"],
            "rec": res["recall"],
            "f1": res["f1"],
            "mean_iou": res["mean_iou"],
            "start_err": res.get("mean_start_err_s"),
            "end_err": res.get("mean_end_err_s"),
        })

    # Print markdown table
    print("\n| Video ID | Kind | Gold | Pred | TP | FP | FN | Precision | Recall | F1 | Mean IoU | |start|s | |end|s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "status" in r:
            print(f"| `{r['url']}` | {r['kind']} | - | - | - | - | - | - | - | MISSING | - | - | - |")
        else:
            se = "-" if r.get("start_err") is None else f"{r['start_err']:.2f}"
            ee = "-" if r.get("end_err") is None else f"{r['end_err']:.2f}"
            print(
                f"| `{r['url']}` | {r['kind']} | {r['gold']} | {r['pred']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                f"{r['prec']:.2f} | {r['rec']:.2f} | {r['f1']:.2f} | {r['mean_iou']:.2f} | {se} | {ee} |"
            )

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) else 0.0
    macro_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    print(f"\n--- Aggregate Summary ---")
    print(f"Total Gold Segments: {total_gold} | Total Predicted Segments: {total_pred}")
    print(f"Micro Precision: {micro_prec:.3f} | Micro Recall: {micro_rec:.3f} | Micro F1: {micro_f1:.3f}")
    print(f"Average IoU (True Positives): {macro_iou:.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ad detection predictions against ground truth.")
    parser.add_argument("path", help="Prediction JSON file path OR directory with predictions")
    parser.add_argument("url_substr", nargs="?", help="URL substring to match in ground_truth.json (if single file)")
    parser.add_argument("--all", action="store_true", help="Evaluate all predictions in directory against ground truth")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for true positive match (default: 0.5)")
    args = parser.parse_args()

    p = Path(args.path)
    if args.all or p.is_dir():
        evaluate_all(p, args.iou)
    elif args.url_substr:
        match_video(p, args.url_substr, args.iou)
    else:
        # Infer url_substr from prediction json if possible
        data = json.loads(p.read_text(encoding="utf-8"))
        url = data.get("source", {}).get("url", "")
        if url:
            vid_id = video_id_from_url(url)
            match_video(p, vid_id, args.iou)
        else:
            print("Error: Please provide url_substr or a prediction file containing source.url")
            sys.exit(2)