# Ad segment detection

Take-home detector: URL or local file in → JSON timeline of advertising segments out.

Policy, signals, and evaluation live in `DESIGN.md` and `EVAL.md`. Hand labels: `data/ground_truth.json`.

## Install (clean machine)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# system deps: ffmpeg, tesseract
# Debian/Ubuntu: sudo apt install ffmpeg tesseract-ocr
```

Optional hosted judge later: export `OPENAI_API_KEY` or `GEMINI_API_KEY`. v0.1 is local OCR + rules and does not need a key.

## Run the API

```bash
cd ad-segment-detector
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/detect \
  -H 'content-type: application/json' \
  -d '{"url":"https://www.youtube.com/shorts/Ve0zdhTQA4U","kind_hint":"short"}'
```

Local file (Reels, live recording):

```bash
curl -s localhost:8000/v1/detect \
  -H 'content-type: application/json' \
  -d '{"local_path":"/abs/path/reel.mp4","kind_hint":"short"}'
```

## CLI

```bash
PYTHONPATH=. python -m app.main --local /abs/path/short.mp4 --kind short --out /tmp/out.json
PYTHONPATH=. python scripts/eval.py /tmp/out.json Ve0zdhTQA4U
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests -q
```

## Instagram Reels

Do not scrape. Capture the mp4 yourself and pass `local_path`. Production acquisition notes are in `DESIGN.md`.

## Test set

1. `https://www.youtube.com/watch?v=ujFWRFYLGjY` — long-form VOD  
2. `https://www.youtube.com/shorts/Ve0zdhTQA4U` — Short  
3. `https://www.youtube.com/watch?v=s0LLVQeMmtU` — live (or a 20+ min recording of the same channel)  
4. `https://www.instagram.com/reel/DJl6-v8oufg` — local file  
5. `https://www.instagram.com/reel/Db78RoIuOyl/` — local file  
