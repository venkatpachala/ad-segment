# EVAL.md

Detection accuracy is an input to this write-up, not a score. The interesting part is what we labelled, how we scored, and why three things fail.

## 1. Ground-truth methodology

I labelled all five assignment URLs myself against the policy in `DESIGN.md`. Procedure:

1. Watch once without the detector.
2. Mark every contiguous span whose *primary purpose* is commercial.
3. Record brand, type, and the frames/transcript that would justify the call.
4. Write rejected near-misses (network bugs, comedy, coat logos) so a second labeler can disagree in the open.

Labels: `data/ground_truth.json` (same schema as the API, plus an `annotation` block). Extra fields are allowed by the contract.

**Live input.** `https://www.youtube.com/watch?v=s0LLVQeMmtU` was offline / YouTube bot-checked from this IP on 28 Aug 2026. I processed a **662s recording of the same channel** (`data/cache/s0LLVQeMmtU.mp4`) through `POST /v1/live/sessions` / `python -m app.live_main`. Timestamps are relative to that file, not to wall-clock IST. A third commercial I had noted on a longer viewing (~15:03) is **not in the 662s file**, so it is not gold.

| ID | Kind | Duration | Gold | Why |
|---|---|---|---|---|
| ujFWRFYLGjY | vod | 2806s | 0.0–7.0 preroll, Vista Imaging | Dedicated opening commercial before podcast intros. |
| Ve0zdhTQA4U | short | 63.9s | 5.0–63.0 overlay, Avis Vascular | Banner with phone; educational speech is not the ad. |
| s0LLVQeMmtU | live | 662s recording | 222–252, 497–525 full-screen breaks | Channel bug rejected (R-Logo). |
| DJl6-v8oufg | short | 13.0s | 8.0–10.0 self_promo end card | Tutorial body is not an ad; contact card is. |
| Db78RoIuOyl | short | 22.6s | **none** | Clinic comedy about cap prices. No CTA, no disclosure. |

Ambiguous cases I resolved in the labels: coat logos = non-ad; Avis fade in/out = one overlay; comedy account type ≠ ad.

## 2. Metrics

Segment-level, not frame accuracy.

- **IoU** of `[start_s, end_s]` vs gold.
- **TP** if greedy 1–1 match has IoU ≥ 0.5.
- **FP / FN** as usual. Empty-vs-empty (comedy reel) is P=R=1.
- **Mean IoU** over TPs.
- **Boundary error** (reported for TPs): mean \|start_pred − start_gold\| and \|end_pred − end_gold\|.

```bash
python scripts/eval.py data/eval --all --iou 0.5
```

## 3. Per-video results

Frozen predictions in `data/eval/`. Rule judge, no LLM (`model_calls=0`, `estimated_cost_usd=0`).

| Video | Kind | Gold | Pred | TP | FP | FN | P | R | F1 | Mean IoU | Boundary (s) | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ujFWRFYLGjY | vod | 1 | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.0 / 0.0 | 22.3 |
| Ve0zdhTQA4U | short | 1 | 1 | 1 | 0 | 0 | 1.00 | 1.00 | 1.00 | 0.98 | 0.5 / 0.9 | 17.9 |
| s0LLVQeMmtU | live | 2 | 0 | 0 | 0 | 2 | 0.00 | 0.00 | 0.00 | — | — | 179 |
| DJl6-v8oufg | short | 1 | 1 | 0 | 1 | 1 | 0.00 | 0.00 | 0.00 | 0.17* | — | 19.3 |
| Db78RoIuOyl | short | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | — | 14.2 |

\*IoU of the unmatched pair, shown for diagnosis; it is **not** a TP.

**Micro (five videos, 5 gold spans):** TP=2, FP=1, FN=3 → P=0.67, R=0.40, F1=0.50.

Type/brand on the two TPs is correct (Vista preroll; Avis overlay). Overlay evidence law holds: empty `transcript_span`, banner in `ocr_text`.

## 4. Three failure cases

### F1 — Live commercial breaks, 0 committed segments (both gold FN)

**Symptom.** 331 ticks, 662s, jsonl honest (`now_s` starts at 2; `now=40` has no t=400). Final `segments: []`. Gold 222–252 and 497–525 missed.

**Root cause.** Live sensors OCR two *corners* and require the **same** brand-like string to persist **≥8s**. Asianet-style breaks in this recording are full-screen spots that cut every 1–3s. Nothing stable sits in TR/BR except the channel bug and ticker, which the chrome stoplist / Latin-token gate correctly drop. Scene-cut proposer exists but does not promote “any hard cut” to an ad — that would label every news package.

**Why the design still looks like this.** The opposite bug (ticker → fake overlays) is worse on news. We chose precision on chrome and accepted recall loss on slates.

**Fix with two more weeks.** A full-frame slate classifier fired only on scene cuts, committed on a run of cuts without news-anchor faces. Not a bigger ROI.

### F2 — Dental Reel end card expanded to the whole clip (IoU 0.17)

**Gold.** 8.0–10.0s self_promo (name, address, phone after a silent brush demo). Duration 13s.
**Pred.** 1.0–12.98s `other` overlay, brand correct (`Asian Dental / Dr. Samskruti`).

Intersection 2s / union 12s → IoU 0.17 < 0.5 → counted as FP+FN.

**Root cause.** OCR sees the clinic name for most of the Reel (logo on the coat / lower third), and the overlay policy says: persistent brand+CTA → first frame to last frame. That policy is right for an all-ad Reel (ambiguity 7) and wrong for a tutorial with a late contact card (ambiguity 7’s inverse). The judge never saw a reason to *shrink* to the last 2s.

**Fix.** Require phone/address density, not just brand word, before extending to `[0, T]`; otherwise snap to the last high-score OCR cluster.

### F3 — This test set has no spoken mid-roll (blind spot, not a scored FN)

The long-form gold is a 7s opening slate. Item 6 of the brief — 90s sponsor read, picture unchanged — is the job ASR/intent were built for, and it does not appear in the five URLs. I will not invent a mid-roll F1. If a reviewer plays a creator-read VPN spot, the live/VOD speech path is the one to watch, not OCR.

A related small miss: VOD body sampling is 0.25 fps after 10s, so a 1.4s mid-video bumper (ambiguity 3) would be a FN by construction. None of the five golds is that bumper.

## 5. Honesty checks (live)

On `data/eval/s0LLVQeMmtU.jsonl` (same run):

- First event `now_s` ≈ 2.
- No event with `now_s=40` contains a span near 400.
- `asianet` / clock / LIVE not committed.
- `lookahead` is false; ticks written while the loop ran, not after EOF.

## 6. Determinism

These numbers are from one rule-judge pass. Re-runs on the same local files move Short bounds by at most one 0.5s sample. Enabling the LLM will jitter bounds 1–3s; say so in the interview.

## 7. Cost of the test set

**$0.00.** 0 model calls. Reviewer can reproduce without a key. If they set `OPENAI_API_KEY`, skip still applies to OCR-only overlays; a speech candidate would cost ~$0.002.

## 8. What I would not claim

70% of *these* gold spans is not 70% of “ads in the wild.” Two of five videos are overlays the OCR path likes; live news slates are the failure the architecture predicted; the end-card Reel is a policy disagreement, not a sensor miss.
