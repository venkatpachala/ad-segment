# DESIGN.md — Ad segment detection

This system takes a video URL or a local file and returns a JSON timeline of advertising segments. Two HTTP surfaces, one policy:

- `POST /v1/detect` — Shorts and VOD. Kind is inferred (`short` if duration ≤ 180s, else `vod`). Upload is the SLA; YouTube/Instagram URL fetch is best-effort.
- `POST /v1/live/sessions` — live ticks until stop. Horizon 0: at media time `t` the detector may only use `[0, t]`.

Instagram is an ingest dialect, not a fourth detector. Reels use the Short funnel. Instagram Live and Stories are out of v1.

## 1. What counts as an ad

An **advertising segment** is a contiguous stretch of the *downloaded media* whose **primary purpose** is to commercially promote, sell, or drive an action toward a product, service, brand, paid relationship, or the creator’s own offering.

It is an ad when at least one of these holds:

- Spoken pitch with a join / buy / enroll / discount-code speech act.
- A dedicated commercial slate or bumper before editorial content.
- An on-screen offer (brand + phone, price, or CTA) burned into the pixels.
- An explicit sponsorship or affiliate disclosure that occupies a timed block.

It is **not** an ad when:

- The host greets the channel, asks for likes, or names the show.
- A logo, merch hoodie, or network bug is visible with no offer.
- Third-party banners appear inside a webpage the host is reviewing.
- Platform pre-roll never lands in the file we downloaded.
- Comedy or news *mentions* a price without a commercial ask.

Types we emit (contract enum): `preroll`, `midroll_sponsor_read`, `product_placement`, `self_promo`, `affiliate`, `platform_inserted`, `bumper`, `other`. We use `other` for burned-in overlays (hospital L-bar, clinic banner). `platform_inserted` is reserved for ads that exist in the file as a distinct clip; we do not invent it for YouTube’s unseen pre-roll.

**Evidence law.** OCR-only overlay → `signals_used=["ocr"]`, empty `transcript_span`, banner in `ocr_text`, brand from OCR. Speech evidence is included only if ASR or intent actually fired. Captions prefer `*-orig` tracks, never auto-translated Hindi.

## 2. Ambiguity pack

1. **“Join my Patreon.”** Not an ad. Channel support is community, not a timed commercial block, unless it becomes a scripted mid-roll with a unique URL and a hard cut back to content.
2. **Hoodie with own merch logo the whole video.** Not an ad. Continuous identity, no offer, no discrete start. Same ruling as a network bug.
3. **1.4s “sponsored by” bumper.** This is an ad (`bumper`), and it is why uniform 1 fps is illegal. A 1.4s card sampled at 1 Hz is missed with probability ~0.6. Shorts sample at 2 fps; VOD opens with a 2 fps burst for 10s then 0.25 fps. We still miss a mid-body 1.4s bumper after second 10. That is a stated miss, not an accident.
4. **40s of an official trailer inside a review.** Not an ad. Editorial fair-use of the work being reviewed. It would be an ad only with a buy/rent CTA for a different title, or a sponsor read around it.
5. **Platform pre-roll that never appears in the download.** Out of scope. We label the file we ingested. `platform_inserted` is unused unless the bytes contain it. YouTube client-side ads are a different product.
6. **90s sponsor read, picture never changes.** Audio is the signal. Frames at 0.25 fps will look identical (MAE skip will refuse to re-OCR). ASR + intent cosine on 25s speech windows is the proposer; the LLM/rule judge tightens bounds from the transcript. Sampling frames denser does not help.
7. **Reel that is an ad from frame one to the last frame.** `start_s = 0`, `end_s = duration`. One segment. We do **not** invent a non-ad body that is not there. (Contrast: a tutorial with a 2s end card is only the card.)
8. **Two sponsors back to back, no gap.** Two segments if brands differ; one if it is the same read with a second product. Split on brand change, not on a breath.

Items 3 and 6 drove the sampler: dense where visual ads are short and front-loaded; sparse where speech carries the mid-roll.

## 3. Architecture and signals

```
URL or upload
    │
    ▼
ingest (yt-dlp / file) ── captions *-orig (parallel)
    │
    ▼
sample frames ── scene cuts (skip if duration < 60s)
    │
    ▼
OCR (MAE skip, 4 workers)     ASR windows     intent BoW
    │                              │               │
    └──────────────┬───────────────┴───────────────┘
                   ▼
            union + drop incidental OCR
                   ▼
         judge: OpenRouter → OpenAI gpt-4o-mini → rules
                   ▼
            snap to past scene cuts → JSON
```

**Why these signals, not a VLM on every frame.** In long-form Indian YouTube, the mid-roll is almost always spoken. In Shorts/Reels in this test set, the ad is a burned-in banner. Live news ads in this capture are full-screen slates or L-bars. A vision-LLM at 1 fps would cost dollars per hour and still miss the 90s read with a static frame (item 6). Tesseract is the right cheap sensor for overlays; captions/Whisper for speech; the judge only sees shortlisted packets.

Live is the same detectors on a 2s media clock, not VOD on a finished file:

```
tick every 2s of media
  sensors on [now-16s, now]
  ROI OCR: top-right and bottom-right 28%×22%
  persist ≥8s same text in same ROI → provisional
  drop <3s flashes
  commit after 6s silence or 180s max
  judge runs on commit only
  end_s is always ≤ now_s
```

A sponsor read that straddles a tick stays in `open` with a frozen `start_s` and a growing `end_s`. We do not wait for future audio to “confirm the end.” Commit is silence, not lookahead.

## 4. Instagram in production (not implemented)

v1 accepts a public Reel URL through yt-dlp plus a Netscape cookie jar at `data/instagram.cookies.txt`, and treats **file upload as the production guarantee**. Automated acquisition is not scored.

If this had to run at production volume I would not scrape:

- **Auth.** Instagram Graph API / official partner feeds for accounts you own or have licensed. End-user OAuth if the user is depositing their own Reel. No shared consumer cookies in a server.
- **Rate limits.** Graph API is on the order of hundreds of calls per hour per token. A queue with per-token budgets, backoff on 429, and a cache keyed by shortcode. Burst scraping is how IPs get banned.
- **Legal.** Instagram’s terms prohibit crawling and circumventing technical measures. A downloaded Reel for a take-home eval, acquired by hand, is a local file we process. Shipping a bot that logs into Instagram to defeat anti-automation is both out of scope and a liability. Production copy must come from a licensed source or the uploader.

## 5. What I tried that did not work

- **Uniform 1 fps everywhere.** Misses item-3 bumpers and Short overlay fade-ins. Replaced with kind-specific sampling.
- **Full-frame OCR every live tick.** The news ticker and “LIVE” chrome dominate Tesseract. Switched to TR/BR ROI, chrome stoplist, Latin token ≥4 chars, MAE skip. Cost of that fix: full-screen slates that do not sit in a corner are invisible (see EVAL live miss).
- **Trusting YouTube’s translated `hi` captions.** Telugu videos produced Hindi evidence. Now `*-orig` only.
- **Treating OpenRouter 429 as a failed job.** Overlay prerolls do not need an LLM. Skip LLM when the packet is OCR-only; retry/jitter; then OpenAI; rules last. `stats.llm_fallback` records what happened.
- **MiniLM embeddings for intent.** Cold-start and RAM cost for a take-home. Bag-of-words cosine against promo vs anti-promo vectors is <1ms and enough to propose.

## 6. Cost, latency, how they scale

Numbers from the frozen eval runs (local Tesseract, LLM skipped):

| Video | Duration | Wall | × realtime | Frames OCR’d | Model calls | USD |
|---|---|---|---|---|---|---|
| VOD ujFWRFYLGjY | 2806s | 22.3s | 0.008× | 78 | 0 | 0 |
| Short Ve0zdhTQA4U | 64s | 17.9s | 0.28× | 28 | 0 | 0 |
| Live recording | 662s | 179s | 0.27× | 548 | 0 | 0 |
| Reel DJl6-v8oufg | 13s | 19.3s | 1.5× | 26 | 0 | 0 |
| Reel Db78RoIuOyl | 23s | 14.2s | 0.63× | 19 | 0 | 0 |

Test-set spend: **$0.00**. Hosted judge, when used, is billed at ~$0.002/call in `stats.estimated_cost_usd`. A 46-minute VOD at 0.008× realtime is CPU (Tesseract + yt-dlp), not GPU. Dollars per hour of video on the rule path: ~$0. On gpt-4o-mini with one batch call per video: ~$0.002–0.01 per hour of video.

Live fake-live CLI races the OCR; a real HLS session ticks every 2s of wall and must stay under ~1.8s of work (skip Whisper and cut-OCR on overrun). HLS glass-to-detect is 8–25s from GOP+CDN; we do not claim frame-accurate on-air time.

To get 10× cheaper: drop VOD body OCR entirely and keep opening burst + ASR. Recall cost: overlay-only mid-rolls vanish; spoken sponsor reads remain.

## 7. What breaks at 1,000 videos/day

Tesseract is the bottleneck. At ~20–40s CPU per video, 1,000 videos is ~8–11 CPU-hours — one beefy box if queued, not if they all arrive at 09:00. Failures that show up before CPU does:

- yt-dlp IP reputation and bot-check. Upload-first is the only honest SLA.
- Cookie jars are not a fleet strategy (Instagram especially).
- Live is 1:1 with the stream: 1,000 concurrent lives is 1,000 open `VideoCapture`s and a different architecture (HLS workers, ring buffers).
- A hosted LLM on every candidate would be the first invoice shock; the skip-for-OCR-only path is what keeps the test set under $10.

## 8. Two weeks next

1. A live **slate / scene-cut commercial-break** proposer. Today we require 8s of stable corner text. The Asianet recording’s breaks are full-screen spots that change every couple of seconds — that is the entire live miss.
2. Overlay vs end-card duration: persistent logo on a tutorial is not `[0, T]`; a true all-ad Reel is. The judge prompt says the former; the end-card Reel in the test set is the latter’s inverse and we over-extended.
3. Ten more hours of labeled live news and a second long-form with a spoken mid-roll, which this five-video set does not contain.

## 9. What I cut and why

Cut brand-to-product-page linking (Tier 3). Cut defeating Instagram anti-automation (zero points, eats the week). Cut login, database, Kubernetes. Cut per-frame VLMs. Cut live Whisper by default. Cut a 20+ minute live capture when the stream was offline — shipped 662s of the same channel and said so. Kept the viewer under an hour. Kept evaluation.

SSE on `/v1/live/sessions/{id}/stream` shipped because the live surface already needed an event pump; it was not a polish pass.

## 10. Determinism and AI tools

OCR + rule judge on the same file is deterministic to ±1 sample (0.5s on Shorts). LLM judging is not: expect 1–3s boundary jitter and occasional type flips. The submitted eval numbers are the rule path (`llm_fallback: skipped`).

**AI tools.** Copilot/Grok drafted boilerplate (FastAPI wiring, pytest shells, first-pass docs). I wrote or rewrote the labelling policy, evidence law, live state machine, ROI/MAE OCR, ingest error codes, ground truth, and the eval/failure write-up. I can change any of it live.

## 11. Public API (frozen)

`POST /v1/detect` JSON `{url, file}` or multipart `file`. Finishes in 15s → 200; else 202 `{job_id}` and poll `GET /v1/jobs/{id}`. Live: `POST /v1/live/sessions` then `GET .../events?after=` or SSE, `POST .../stop`. Errors are `{code, detail}` (`youtube_bot_check`, `instagram_auth`, `no_tesseract`, `unsupported`) — never a traceback.
