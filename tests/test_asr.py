from pathlib import Path

from app.pipeline.asr import (
    parse_vtt,
    refine_asr_start,
    score_transcript,
    candidates_from_asr,
    CaptionLine,
    language_from_caption_path,
    select_caption_track,
)
from app.pipeline.candidates import drop_incidental_ocr
from app.pipeline.types import Candidate


def test_hindi_course_pitch_scores():
    text = "हमारे कोर्सेज के बाद कई लोग जॉब प्राप्त कर चुके हैं और इनके टेस्टिमोनियल्स"
    score, trig = score_transcript(text)
    assert score >= 2
    assert trig


def test_hindi_join_program_scores():
    text = "तो आप अगर अपनी जर्नी स्टार्ट करना चाहते हैं इन द फील्ड ऑफ डाटा एनालिटिक्स तो बिल्कुल जॉइ करिए यह प्रोग्राम"
    score, _ = score_transcript(text)
    assert score >= 2


def test_news_reading_does_not_score():
    score, _ = score_transcript("आज की खबरों में मंत्री ने बयान दिया और अखबार में लेख छपा")
    assert score < 2


def test_vtt_parse_and_candidate():
    vtt = """WEBVTT

00:07:48.000 --> 00:07:52.000
हमारे कोर्सेज के बाद कई लोग जॉब प्राप्त कर चुके हैं

00:09:49.000 --> 00:09:54.000
जॉइ करिए यह प्रोग्राम
"""
    lines = parse_vtt(vtt)
    assert len(lines) == 2
    cands = candidates_from_asr(lines, duration_s=621.0)
    assert len(cands) == 1
    cands[0] = refine_asr_start(cands[0], lines)
    assert 468 <= cands[0].start_s <= 472
    assert "कोर्सेज" in " ".join(cands[0].texts)


def test_incidental_mid_vod_ocr_dropped():
    webpage = Candidate(120, 128, "ocr", ["price"], ["BUY XYZ 50% OFF"], [120], 4)
    spoken = Candidate(469, 621, "asr", ["strong:कोर्सेज"], ["हमारे कोर्सेज"], [], 6)
    kept = drop_incidental_ocr([webpage, spoken], kind="vod", duration_s=621)
    assert all("asr" in c.source or c.start_s < 15 for c in kept)
    assert spoken in kept
    assert webpage not in kept


_TE_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
నమస్కారం ఈ వీడియోలో
"""

_HI_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
नमस्कार यह अनुवाद है
"""

_EN_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello and welcome to the video
"""


def test_prefers_te_orig_over_translated_hi(tmp_path: Path):
    (tmp_path / "captions.hi.vtt").write_text(_HI_VTT, encoding="utf-8")
    (tmp_path / "captions.te-orig.vtt").write_text(_TE_VTT, encoding="utf-8")
    chosen = select_caption_track(list(tmp_path.glob("*.vtt")))
    assert chosen is not None
    assert "te-orig" in chosen.name
    assert language_from_caption_path(chosen) == "te"
    lines = parse_vtt(chosen.read_text(encoding="utf-8"), lang="te")
    assert lines
    assert "నమస్కారం" in lines[0].text
    assert lines[0].lang == "te"


def test_english_video_picks_en_not_hi(tmp_path: Path):
    (tmp_path / "captions.hi.vtt").write_text(_HI_VTT, encoding="utf-8")
    (tmp_path / "captions.en.vtt").write_text(_EN_VTT, encoding="utf-8")
    chosen = select_caption_track(list(tmp_path.glob("*.vtt")))
    assert chosen is not None
    assert language_from_caption_path(chosen) == "en"
    lines = parse_vtt(chosen.read_text(encoding="utf-8"), lang="en")
    assert "Hello" in lines[0].text
    assert lines[0].lang == "en"