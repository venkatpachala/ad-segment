# Ad segment detection

URL or local file in → JSON timeline of advertising segments out.

Policy: `DESIGN.md`. Numbers: `EVAL.md`. Labels: `data/ground_truth.json` (copy at repo root `ground_truth.json`).

Two endpoints. Kind is never sent by the client.

| Surface | When |
|---|---|
| `POST /v1/detect` | Short or VOD. `short` if duration ≤ 180s, else `vod`. |
| `POST /v1/live/sessions` | Live. Ticks until `POST .../stop`. Horizon 0. |

Upload / `--local` is the SLA. YouTube and Instagram URL fetch is best-effort (bot-check and login walls are expected).

## 10-minute clean-machine path

Needs Python 3.12+, ffmpeg, Tesseract (English; Telugu tessdata optional but used on this set).

```bash
# Debian/Ubuntu
sudo apt install -y ffmpeg tesseract-ocr tesseract-ocr-tel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (winget or chocolatey). ffmpeg + tesseract on PATH.
# If you already have the portable tree at .\tesseract\tesseract.exe, that is picked up automatically.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional hosted judge (not required for the eval numbers):

```bash
cp .env.example .env
# OPENROUTER_API_KEY and/or OPENAI_API_KEY
```

No key → rule judge, `$0`. Never commit `.env`.

```bash
python -m pytest tests -q
# expect 88 passed

python -m uvicorn app.main:app --port 8000
# other terminal:
curl -s localhost:8000/health
# {"ok": true, "tesseract": true, ...}
```

Viewer (one HTML page, seek-bar overlays): [http://localhost:8000/viewer/](http://localhost:8000/viewer/) — paste a local mp4 and the detect JSON, click Draw.

## Reproduce the eval (local files)

Predictions used in `EVAL.md` are already in `data/eval/`. To re-run:

```bash
# Short (Avis overlay) — fixture shipped
python -m app.main --local data/fixtures/Ve0zdhTQA4U.mp4 --out data/eval/Ve0zdhTQA4U.json

# Instagram Reels — fixtures shipped (do not scrape)
python -m app.main --local data/fixtures/ig_DJl6-v8oufg.mp4 --out data/eval/DJl6-v8oufg.json
python -m app.main --local data/fixtures/ig_Db78RoIuOyl.mp4 --out data/eval/Db78RoIuOyl.json

# Long-form VOD — download once, then --local (file is >100MB, not in git)
python -m app.main --url "https://www.youtube.com/watch?v=ujFWRFYLGjY" --out data/eval/ujFWRFYLGjY.json
# on youtube_bot_check: save the mp4 yourself, then:
# python -m app.main --local path/to/ujFWRFYLGjY.mp4 --out data/eval/ujFWRFYLGjY.json

# Live — MUST use the live CLI, not app.main. Stream was offline; use a recording of the same channel.
python -m app.live_main --local path/to/s0LLVQeMmtU.mp4 --out data/eval/s0LLVQeMmtU.jsonl --out-json data/eval/s0LLVQeMmtU.json

python scripts/eval.py data/eval --all --iou 0.5
```

API equivalents:

```bash
curl -s localhost:8000/v1/detect -F file=@data/fixtures/Ve0zdhTQA4U.mp4
# 200 body or 202 {"job_id","status"} then:
# curl -s localhost:8000/v1/jobs/JOB_ID

curl -s localhost:8000/v1/live/sessions -F file=@path/to/s0LLVQeMmtU.mp4
# {"session_id","status":"running"}
curl -s "localhost:8000/v1/live/sessions/SID/events?after=0"
curl -s -X POST localhost:8000/v1/live/sessions/SID/stop
```

Cookies if you insist on URL ingest (optional, separate jars):

- `data/youtube.cookies.txt`
- `data/instagram.cookies.txt`

Netscape format. Never commit them.

## Output contract

Required keys: `source.{url,platform,kind,duration_s,processed_at}`, `segments[]` with `id,start_s,end_s,ad_type,confidence,brand,description,evidence.{frame_timestamps,transcript_span,signals_used}`, `stats.{wall_clock_s,estimated_cost_usd,frames_sampled,model_calls}`. Extra fields (`ocr_text`, `presentation`, `llm_fallback`) are present and allowed.

## Test set (do not substitute)

1. https://www.youtube.com/watch?v=ujFWRFYLGjY
2. https://www.youtube.com/shorts/Ve0zdhTQA4U
3. https://www.youtube.com/watch?v=s0LLVQeMmtU — live path; recording if offline
4. https://www.instagram.com/reel/DJl6-v8oufg — local file
5. https://www.instagram.com/reel/Db78RoIuOyl/

## 5-minute recording

Script: `RECORDING.md`. Link the video in the email thread with the private repo URL.
