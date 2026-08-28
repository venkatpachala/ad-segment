"""Instagram is an ingest dialect. No network in these tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.ingest import (
    IngestError,
    classify_instagram_error,
    classify_url,
    infer_kind,
    instagram_shortcode,
    is_instagram_url,
    is_live_url,
    normalize_instagram_url,
)


def test_ig_url_parser_reel_and_share():
    reel = "https://www.instagram.com/reel/DJl6-v8oufg/?igsh=abc&utm_source=ig"
    assert is_instagram_url(reel)
    assert instagram_shortcode(reel) == "DJl6-v8oufg"
    norm = normalize_instagram_url(reel)
    assert "igsh" not in norm
    assert "utm_" not in norm
    assert "/reel/DJl6-v8oufg" in norm

    share = "https://www.instagram.com/share/reel/DJl6-v8oufg"
    assert instagram_shortcode(share) == "DJl6-v8oufg"
    assert infer_kind(share, None) == "short"

    bare = "https://instagram.com/reels/XXXX1234yyyy"
    assert instagram_shortcode(bare) == "XXXX1234yyyy"


def test_ig_p_and_tv_shortcode():
    assert instagram_shortcode("https://www.instagram.com/p/AbCdEfGhIjK/") == "AbCdEfGhIjK"
    assert instagram_shortcode("https://www.instagram.com/tv/AbCdEfGhIjK/") == "AbCdEfGhIjK"
    # /p/ without duration is vod until ffprobe; duration wins after probe
    assert infer_kind("https://www.instagram.com/p/AbCdEfGhIjK/", None) == "vod"
    assert infer_kind("https://www.instagram.com/p/AbCdEfGhIjK/", None, duration_s=52.0) == "short"
    assert infer_kind("https://www.instagram.com/p/AbCdEfGhIjK/", None, duration_s=400.0) == "vod"


def test_classify_instagram_platform():
    platform, kind = classify_url(
        "https://www.instagram.com/reel/DJl6-v8oufg/", None, None
    )
    assert platform == "instagram"
    assert kind == "short"


def test_ig_live_is_not_youtube_live_surface():
    ig_live = "https://www.instagram.com/channel/live/123"
    assert is_instagram_url(ig_live)
    assert not is_live_url(ig_live)
    yt_live = "https://www.youtube.com/live/s0LLVQeMmtU"
    assert is_live_url(yt_live)


def test_classify_instagram_errors():
    assert classify_instagram_error("Login required to download") == "instagram_auth"
    assert classify_instagram_error("HTTP Error 429 Too Many Requests") == "rate_limited"
    assert classify_instagram_error("This post does not contain a video") == "unsupported_media"
    assert classify_instagram_error("Requested content is not available (404)") == "private_or_missing"


def test_local_reel_file_uses_short_funnel(tmp_path: Path):
    src = Path("data/video1_0_15.mp4")
    if not src.exists():
        pytest.skip("no local short fixture")
    from app.pipeline.ingest import ingest

    asset = ingest(
        "https://www.instagram.com/reel/DJl6-v8oufg/",
        str(src),
        None,
        tmp_path,
    )
    assert asset.platform == "instagram"
    assert asset.kind == "short"
    assert asset.duration_s <= 180
    assert asset.path.exists()


def test_ig_stories_rejected_without_file(tmp_path: Path):
    from app.pipeline.ingest import ingest

    with pytest.raises(IngestError) as ei:
        ingest(
            "https://www.instagram.com/stories/someone/123/",
            None,
            None,
            tmp_path,
        )
    assert ei.value.code == "unsupported"
