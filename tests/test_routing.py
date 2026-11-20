"""Tests for the OCR-first router and OCR reading-order helper (no Ollama/OCR engine needed)."""

from __future__ import annotations


def test_plain_text_routes_to_text():
    from quizai.routing import decide_route

    d = decide_route("What is 25% of 200? a) 25 b) 50 c) 75", 0.97, 30)
    assert d.use_vision is False


def test_referential_phrase_routes_to_vision():
    from quizai.routing import decide_route

    for text in [
        "Refer to the figure above. What is the slope?",
        "Which graph represents f(x)=x^2?",
        "In the circuit shown, find the resistance.",
        "Based on the diagram below, name the part.",
    ]:
        assert decide_route(text, 0.99, 40).use_vision is True, text


def test_rendered_math_glyphs_route_to_vision():
    from quizai.routing import decide_route

    # Multiple math glyphs OCR tends to corrupt → vision.
    assert decide_route("Evaluate ∫ x² dx and √144", 0.95, 25).use_vision is True


def test_single_superscript_routes_to_vision():
    from quizai.routing import decide_route

    # A single x² is enough: OCR often flattens it to x2 even at high confidence.
    assert decide_route("Solve x² - 5x + 6 = 0", 0.99, 22).use_vision is True


def test_visual_noun_without_spaces_routes_to_vision():
    from quizai.routing import decide_route

    # OCR merges words: "the chart" becomes "thechart" — still must route.
    assert decide_route("Accordingtothechart,whatwasrevenue?", 0.99, 36).use_vision is True


def test_chinese_visual_reference_routes_to_vision():
    from quizai.routing import decide_route

    assert decide_route("如图所示，求阴影部分的面积。", 0.99, 14).use_vision is True
    assert decide_route("根据下表数据计算平均值。", 0.99, 12).use_vision is True


def test_single_stray_glyph_does_not_trip():
    from quizai.routing import decide_route

    # One degree sign alone shouldn't force vision.
    assert decide_route("The angle is 90° — what is its complement?", 0.97, 40).use_vision is False


def test_low_confidence_routes_to_vision():
    from quizai.routing import decide_route

    assert decide_route("some garbled text here", 0.30, 20, conf_floor=0.55).use_vision is True


def test_empty_ocr_routes_to_vision():
    from quizai.routing import decide_route

    assert decide_route("", 0.0, 0).use_vision is True


def test_reading_order_single_column_sorts_by_y():
    from quizai.ocr import OCRLine, _reading_order

    lines = [
        OCRLine("third", 0.9, x=20, y=300),
        OCRLine("first", 0.9, x=22, y=10),
        OCRLine("second", 0.9, x=21, y=150),
    ]
    assert [ln.text for ln in _reading_order(lines)] == ["first", "second", "third"]


def test_reading_order_two_columns_left_then_right():
    from quizai.ocr import OCRLine, _reading_order

    # Left column x~30, right column x~540 — should read all of left, then right.
    lines = [
        OCRLine("L1", 0.9, x=30, y=20),
        OCRLine("R1", 0.9, x=540, y=20),
        OCRLine("L2", 0.9, x=31, y=90),
        OCRLine("R2", 0.9, x=541, y=90),
    ]
    assert [ln.text for ln in _reading_order(lines)] == ["L1", "L2", "R1", "R2"]


# ---------------------------------------------------- cloud direct-vision path
def _cloud_hint_for() -> str:
    """Run the cloud fast path with a fake backend and return the OCR hint."""
    from quizai.app import ApiWorker, _ScreenJob
    from quizai.backends.base import QAItem

    captured: list[str] = []

    class _FakeBackend:
        def answer_screen(self, text: str, png: bytes):
            captured.append(text)
            return [QAItem(question="Q", answer="A", explanation="E")]

    worker = ApiWorker()
    worker._provider = "openai"
    worker._backend = _FakeBackend()
    assert worker._try_fast_path(_ScreenJob(png=b"png"))
    return captured[0]


def test_cloud_capture_never_sends_ocr_text():
    """Cloud providers get the image alone; OCR is skipped entirely."""
    assert _cloud_hint_for() == ""


def test_backend_without_answer_screen_falls_back():
    """Anthropic-style backends fall back to detect+answer without OCR."""
    from quizai.app import ApiWorker, _ScreenJob

    class _FakeBackend:
        def answer_screen(self, text: str, png: bytes):
            raise NotImplementedError

    worker = ApiWorker()
    worker._provider = "anthropic"
    worker._backend = _FakeBackend()
    assert worker._try_fast_path(_ScreenJob(png=b"png")) is False
