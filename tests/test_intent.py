from app.pipeline.asr import CaptionLine
from app.pipeline.intent import candidates_from_intent, intent_score


def test_product_demo_scores_without_brand_list():
    text = (
        "humidity wrecked my hair so i use this range every day here is my routine "
        "shampoo then mask then serum try this link in description"
    )
    score, why = intent_score(text)
    assert score >= 0.18
    assert why


def test_news_and_intro_score_low():
    news, _ = intent_score("today the article says the election commissioner asked about voters")
    intro, _ = intent_score("नमस्कार वेलकम टू करियर 247 मैं हूं प्रशांत धवन आज खबर")
    assert news < 0.18
    assert intro < 0.22


def test_topk_windows_from_mixed_transcript():
    lines = [
        CaptionLine(0, 10, "welcome to my channel today news about elections"),
        CaptionLine(20, 40, "the newspaper reported voters and photo id"),
        CaptionLine(90, 120, "i use this product every day try this range link in description"),
        CaptionLine(150, 170, "like and subscribe comment below"),
    ]
    cands = candidates_from_intent(lines, 180)
    assert cands
    assert any(c.start_s >= 80 for c in cands)
    assert all(c.source == "intent" for c in cands)