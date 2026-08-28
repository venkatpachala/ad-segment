# 5-minute screen recording (script)

Record this once. Talk over the screen. Do not polish the viewer.

**0:00–0:20 — What it is.** Two POSTs. Detect infers short vs vod. Live is a different clock. Upload is the SLA.

**0:20–1:30 — Avis Short (the thing that works).** Play `data/fixtures/Ve0zdhTQA4U.mp4` in `/viewer/`. Paste `data/eval/Ve0zdhTQA4U.json`. Show the red bar ~5–64s, brand Avis, `signals_used: ["ocr"]`, empty `transcript_span`, `ocr_text` with the phone. Say: educational speech is not the ad; the banner is.

**1:30–2:20 — Vista VOD preroll.** Open `data/eval/ujFWRFYLGjY.json`. 0–7s, Vista Imaging, OCR-only. 46 minutes processed in ~22s.

**2:20–3:20 — Live miss (the thing that is wrong).** Open the tail of `data/eval/s0LLVQeMmtU.jsonl`: ticks at 2s, `now_s=40` has no future ads, `segments: []` at the end. Gold 222–252 and 497–525 were full-screen breaks. Detector wants 8s of stable corner text. That is the designed miss.

**3:20–4:20 — End-card miss.** `data/eval/DJl6-v8oufg.json` is 1–13s; gold is 8–10s. IoU 0.17. Clinic name on screen too early, overlay policy extended to EOF.

**4:20–5:00 — Table.** `EVAL.md` micro F1 0.50. Comedy reel correctly empty. Cost $0. Stop.

Do not narrate Instagram scraping. Do not walk through Kubernetes.
