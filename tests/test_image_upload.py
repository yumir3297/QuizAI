"""Tests for image compression on upload and for transient-error retries.

Screenshots are large PNGs; provider CDNs (the edge in front of DeepSeek is
TencentEdgeOne) intermittently reject image bodies with HTTP 550. Images are
therefore re-encoded before they leave the machine, under a 150 KB budget that
keeps diagrams legible while staying in the reliable size range.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

# 150 KB image → ~200 KB base64, plus the JSON envelope.
BODY_BUDGET = 210_000


# ------------------------------------------------------------------ mime sniffing
def test_sniff_png():
    from quizai.backends.base import sniff_image_mime

    assert sniff_image_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"


def test_sniff_jpeg():
    from quizai.backends.base import sniff_image_mime

    assert sniff_image_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"


def test_sniff_webp():
    from quizai.backends.base import sniff_image_mime

    assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"


def test_sniff_unknown_defaults_to_png():
    from quizai.backends.base import sniff_image_mime

    assert sniff_image_mime(b"not an image") == "image/png"


# --------------------------------------------------------------- upload encoding
def _png_bytes(size=(2560, 1600), colour=(30, 120, 200)) -> bytes:
    """A screenshot-like PNG: light background with dense text-like blocks.

    Real screenshots compress far better than random noise, so the fixture
    mimics one (a retina-sized canvas of text lines) instead of using noise.
    """
    import random

    rng = random.Random(1234)
    w, h = size
    img = Image.new("RGB", size, colour)
    px = img.load()
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            px[x, y] = (245, 245, 248)
    for _ in range(600):
        y0 = rng.randrange(0, max(1, h - 20))
        x0 = rng.randrange(0, max(1, w - 200))
        length = rng.randrange(80, max(81, w // 3))
        for y in range(y0, min(y0 + 12, h)):
            for x in range(x0, min(x0 + length, w)):
                px[x, y] = (20, 20, 25)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _noise_png(size=(1600, 1000), seed=7) -> bytes:
    """An incompressible screenshot-like PNG — the worst case for any encoder."""
    import random

    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            px[x, y] = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_small_image_passes_through_untouched():
    from quizai.backends.base import encode_image_for_upload

    data = _png_bytes(size=(50, 50))
    mime, out = encode_image_for_upload(data)
    assert out == data
    assert mime == "image/png"


def test_large_png_is_reencoded_as_jpeg_under_budget():
    from quizai.backends.base import MAX_UPLOAD_BYTES, encode_image_for_upload

    data = _noise_png()
    assert len(data) > MAX_UPLOAD_BYTES

    mime, out = encode_image_for_upload(data)
    assert mime == "image/jpeg"
    assert len(out) <= MAX_UPLOAD_BYTES


def test_reencoded_image_is_still_decodable():
    from quizai.backends.base import encode_image_for_upload

    _, out = encode_image_for_upload(_noise_png())
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"


def test_upload_budget_keeps_request_body_small():
    """The whole point: the encoded body must stay in the CDN's reliable range."""
    import base64

    from quizai.backends.base import encode_image_for_upload

    _, out = encode_image_for_upload(_noise_png())
    body_bytes = len(base64.standard_b64encode(out))
    assert body_bytes < BODY_BUDGET


def test_even_a_noise_screenshot_stays_under_the_cdn_limit():
    """Worst case: an incompressible image must still shrink enough to upload."""
    import base64

    from quizai.backends.base import encode_image_for_upload

    _, out = encode_image_for_upload(_noise_png())
    assert len(base64.standard_b64encode(out)) < BODY_BUDGET


def test_non_image_payload_is_not_dropped():
    from quizai.backends.base import encode_image_for_upload

    data = b"not an image at all" * 5000
    mime, out = encode_image_for_upload(data)
    assert out == data
    assert mime == "image/png"


# ------------------------------------------------------------------ openai backend
class _RecordingHandler(BaseHTTPRequestHandler):
    """Answers with a canned payload; the first N calls drop the connection."""

    replies: list = []
    calls: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).calls.append(body)

        index = len(type(self).calls) - 1
        if index < len(type(self).replies) and type(self).replies[index] is None:
            self.close_connection = True
            self.wfile.close()
            return

        payload = json.dumps(
            {"choices": [{"message": {"content": type(self).content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_server():
    servers = []

    def _start(replies, content):
        handler = type("H", (_RecordingHandler,), {"replies": replies, "calls": [], "content": content})
        srv = HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append((srv, handler))
        return srv, handler

    yield _start

    for srv, _ in servers:
        srv.shutdown()


def test_backend_retries_after_a_dropped_connection(monkeypatch, mock_server):
    import quizai.backends.openai_backend as ob

    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)

    content = json.dumps({"questions": [{"question": "Q", "answer": "A", "explanation": "E"}]})
    srv, handler = mock_server([None], content)

    from quizai.backends import create_backend

    backend = create_backend(
        provider="openai",
        api_key="k",
        model="m",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
    )
    items = backend.answer_screen("ocr text", _png_bytes())

    assert len(handler.calls) == 2  # first attempt dropped, second succeeded
    assert items[0].question == "Q"
    assert items[0].answer == "A"


def test_backend_does_not_retry_a_401(monkeypatch, mock_server):
    """Auth errors must surface immediately instead of burning retries."""
    import urllib.error

    import quizai.backends.openai_backend as ob

    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)

    class _Unauthorized(_RecordingHandler):
        def do_POST(self):
            self.send_response(401)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    srv = HTTPServer(("127.0.0.1", 0), _Unauthorized)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    from quizai.backends import create_backend
    from quizai.backends.base import BackendError

    backend = create_backend(
        provider="openai",
        api_key="k",
        model="m",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
    )
    with pytest.raises(BackendError, match="401"):
        backend.detect_and_extract_question(_png_bytes())
    srv.shutdown()


def test_answer_screen_parses_a_fenced_json_reply(monkeypatch, mock_server):
    import quizai.backends.openai_backend as ob

    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)

    content = '```json\n{"questions":[{"question":"2+2","answer":"4","explanation":"math"}]}\n```'
    srv, _ = mock_server([], content)

    from quizai.backends import create_backend

    backend = create_backend(
        provider="openai",
        api_key="k",
        model="m",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
    )
    items = backend.answer_screen("ocr", _png_bytes())
    assert items[0].question == "2+2"
    assert items[0].answer == "4"


def test_answer_screen_returns_nothing_for_a_non_json_reply(monkeypatch, mock_server):
    import quizai.backends.openai_backend as ob

    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)

    srv, _ = mock_server([], "I cannot help with that.")

    from quizai.backends import create_backend

    backend = create_backend(
        provider="openai",
        api_key="k",
        model="m",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
    )
    assert backend.answer_screen("ocr", _png_bytes()) == []


# --------------------------------------------------- screen prompt / OCR hint
def test_answer_screen_is_image_only_when_ocr_is_empty(monkeypatch, mock_server):
    """Visual/math captures pass no OCR text; the image must be authoritative."""
    import quizai.backends.openai_backend as ob

    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)

    content = json.dumps(
        {"questions": [{"question": "Q", "work": "W", "answer": "A", "explanation": "E"}]}
    )
    srv, handler = mock_server([], content)

    from quizai.backends import create_backend

    backend = create_backend(
        provider="openai",
        api_key="k",
        model="m",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
    )
    items = backend.answer_screen("", _png_bytes(size=(200, 120)))

    body = json.loads(handler.calls[0])
    assert body["temperature"] == 0.0
    text = body["messages"][1]["content"][0]["text"]
    assert "OCR TEXT" not in text
    assert "Read the screenshot directly" in text
    assert items[0].answer == "A"


def test_answer_screen_marks_ocr_as_a_rough_hint(monkeypatch, mock_server):
    import quizai.backends.openai_backend as ob

    monkeypatch.setattr(ob.time, "sleep", lambda _s: None)

    content = json.dumps({"questions": [{"question": "Q", "answer": "A"}]})
    srv, handler = mock_server([], content)

    from quizai.backends import create_backend

    backend = create_backend(
        provider="openai",
        api_key="k",
        model="m",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
    )
    backend.answer_screen("x² = 4", _png_bytes(size=(200, 120)))

    body = json.loads(handler.calls[0])
    text = body["messages"][1]["content"][0]["text"]
    assert "rough hint" in text
    assert "x² = 4" in text
