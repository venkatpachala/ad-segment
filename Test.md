# Benchmark Evaluation Test Set

The evaluation suite tests five distinct video formats to evaluate multimodal ad detection across static VOD, vertical shorts, and zero-lookahead live streams:

1. **Long-Form VOD (YouTube)**: [`https://www.youtube.com/watch?v=ujFWRFYLGjY`](https://www.youtube.com/watch?v=ujFWRFYLGjY)
   - *Format*: VOD (46 min podcast).
   - *Ad Characteristics*: Opening 0.0–7.0s commercial preroll bumper (`Vista Imaging`).

2. **YouTube Short**: [`https://www.youtube.com/shorts/Ve0zdhTQA4U`](https://www.youtube.com/shorts/Ve0zdhTQA4U)
   - *Format*: Short (63.9s health video).
   - *Ad Characteristics*: Persistent burned-in lower-third commercial overlay (`Avis Vascular Center`).

3. **Live Stream / News Broadcast**: [`https://www.youtube.com/watch?v=s0LLVQeMmtU`](https://www.youtube.com/watch?v=s0LLVQeMmtU)
   - *Format*: Live broadcast (evaluated on 662s continuous recording under horizon 0 constraint).
   - *Ad Characteristics*: Full-screen commercial spots & L-bar ads (`VIDA`, `Samsung Galaxy`, `Daikin`).

4. **Instagram Reel (Self-Promo / End Card)**: [`https://www.instagram.com/reel/DJl6-v8oufg`](https://www.instagram.com/reel/DJl6-v8oufg)
   - *Format*: Short Reel (13.0s dental demonstration).
   - *Ad Characteristics*: Closing 8.0–10.0s commercial contact slate (`Asian Dental`).

5. **Instagram Reel (Comedy / Non-Monetized)**: [`https://www.instagram.com/reel/Db78RoIuOyl/`](https://www.instagram.com/reel/Db78RoIuOyl/)
   - *Format*: Short Reel (22.6s dental cap comedy skit).
   - *Ad Characteristics*: Pure narrative comedy skit — **0 gold ad segments** (negative sample).