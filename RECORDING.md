# 5-Minute Screen Recording (Walkthrough Script)

Record this once. Talk over the screen. Keep it crisp, technical, and grounded in code/data.

**0:00–0:30 — Architecture & Surfaces.**
- Two core surfaces: `POST /v1/detect` (static media: VOD vs Shorts/Reels inferred automatically from duration $\le 180\text{s}$) and `POST /v1/live/sessions` (zero-lookahead live stream engine).
- Direct file upload is the production SLA; YouTube/Instagram URL fetch is best-effort.

**0:30–1:30 — Avis Short (Evidence Law & OCR Overlay).**
- Open `http://localhost:8000/viewer/`. Load `data/fixtures/Ve0zdhTQA4U.mp4` and paste `data/eval/Ve0zdhTQA4U.json`.
- Show red bar spanning ~5–64s, brand `Avis Vascular Center`, `signals_used: ["ocr"]`, empty `transcript_span`, `ocr_text` containing phone and CTA.
- Point out: Educational medical speech is editorial; the burned-in lower banner is the advertisement.

**1:30–2:15 — Vista VOD Preroll (Efficiency & Boundary Snapping).**
- Open `data/eval/ujFWRFYLGjY.json`. Show 0.0–7.0s, brand `Vista Imaging`, OCR preroll.
- Highlight performance: 46 minutes of high-res video processed in ~22 seconds ($125\times$ real-time) using adaptive kind-aware sampling.

**2:15–3:30 — Live Stream Zero-Lookahead & Commercial Breaks.**
- Open `data/eval/s0LLVQeMmtU.jsonl` and `data/eval/s0LLVQeMmtU.json`.
- Show live tick structure: clock ticks forward every 2.0s, horizon is strictly enforced (`end_s <= now_s` invariant at all times; at `now_s = 40`, no future knowledge exists).
- Show the 4 committed live ad segments (Hero VIDA at 420s, Samsung Galaxy at 438s, Daikin at 522s, Daikin L-bar at 562s).
- Show how channel bugs (`asianetnews.com`, `LIVE | 07:10 PM`, tickers) are filtered out via Latin token gating and chrome stoplists without false alarms.

**3:30–4:15 — Edge Case: Dental Reel End-Card vs. Full-Clip Logo Expansion.**
- Open `data/eval/DJl6-v8oufg.json`. Show predicted 1.0–12.98s vs gold 8.0–10.0s ($\text{IoU} = 0.17$).
- Explain root cause: Doctor's lab-coat branding was detected in early frames, triggering the full-clip overlay policy (Ambiguity 7). Discuss the two-week fix (requiring dense contact triggers before full-clip extension).

**4:15–5:00 — Benchmark Summary & Cost.**
- Show `EVAL.md` summary table: 7 Gold segments across 5 videos, 6 True Positives, Micro F1 = 85.7%, Mean IoU = 0.996 on True Positives.
- Comedy Reel (`Db78RoIuOyl`) correctly identified as non-commercial ($P = R = 1.0$).
- Total benchmark compute cost: **$0.00** (0 external API calls needed).
- End demo.

