"""OpenAI-compatible backend.

Talks to any server that speaks the OpenAI Chat Completions API over HTTP:

  - OpenAI            (https://api.openai.com/v1)
  - Groq             (https://api.groq.com/openai/v1)
  - OpenRouter       (https://openrouter.ai/api/v1)
  - LM Studio        (http://localhost:1234/v1)
  - any self-hosted OpenAI-compatible server

No third-party SDK required — just stdlib urllib. Vision is sent as a
base64 `data:` image_url, the standard OpenAI multimodal format.

`base_url` selects the server; `api_key` is the Bearer token (local servers
ignore it — a placeholder is fine).
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import TypeVar

from quizai.backends.base import (
    ANSWER_SYSTEM,
    DETECT_SYSTEM,
    AnswerResult,
    Backend,
    BackendError,
    DetectionResult,
    QAItem,
    build_followup_prompt,
    encode_image_for_upload,
    parse_answer_text,
    parse_detection_json,
)
from quizai.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_TIMEOUT_S = 180.0  # local vision servers can take a while on first load

# Transient failures are retried with exponential backoff. The provider CDN
# rejects roughly one upload in five, so a retry usually succeeds.
_MAX_ATTEMPTS = 4
_INITIAL_BACKOFF_S = 1.0

# One-call screen reader: extract every question and answer it in a single
# request (image + OCR text), so a whole capture costs one API call.
_SCREEN_SYSTEM = (
    "You are given a screenshot of a quiz/exam (image) plus OCR text extracted "
    "from it. Extract EVERY question verbatim, including its answer choices, and "
    "answer each. Use the OCR text for exact wording; use the IMAGE to read any "
    "chart, graph, figure, table, or diagram needed to answer. For multiple choice "
    "give the letter AND the option text (e.g. 'D. May'); for numeric give the value. "
    "Add a 1-2 sentence explanation per question.\n"
    'Reply ONLY with JSON: {"questions":[{"question":str,"answer":str,"explanation":str}]}'
)


class OpenAIBackend(Backend):
    """Any OpenAI Chat Completions-compatible endpoint."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        # Allow an empty key (local servers don't need one) — bypass the ABC guard.
        self._api_key = (api_key or "").strip()
        self._model = model or "gpt-4o-mini"
        base = (base_url or "").strip().rstrip("/") or DEFAULT_BASE_URL
        if "://" not in base:
            base = "https://" + base
        self._base_url = base

    # ----------------------------------------------------------------- vision
    def detect_and_extract_question(self, png_bytes: bytes) -> DetectionResult:
        mime, data = encode_image_for_upload(png_bytes)
        b64 = base64.standard_b64encode(data).decode("ascii")
        body = {
            "model": self._model,
            "temperature": 0.0,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": DETECT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Inspect this screenshot. Return the JSON object as specified.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                },
            ],
        }
        return parse_detection_json(self._chat(body))

    # ----------------------------------------------------------------- answer
    def answer_question(self, question: str, context: str | None = None) -> AnswerResult:
        user_text = build_followup_prompt(context, question) if context else question
        body = {
            "model": self._model,
            "temperature": 0.2,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_text},
            ],
        }
        return parse_answer_text(self._chat(body))

    # ------------------------------------------------ one-call screen reader
    def answer_screen(self, ocr_text: str, png_bytes: bytes) -> list[QAItem]:
        """Extract and answer every question on screen in a single request.

        Mirrors the Gemini backend: the OCR text gives exact wording, the image
        covers charts, figures and tables.
        """
        mime, data = encode_image_for_upload(png_bytes)
        b64 = base64.standard_b64encode(data).decode("ascii")
        body = {
            "model": self._model,
            "temperature": 0.2,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": _SCREEN_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "OCR TEXT (exact wording; use the image for "
                                "charts/figures/tables):\n" + (ocr_text or "")
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                },
            ],
        }
        raw = self._chat(body)
        parsed = _parse_json_object(raw)
        if parsed is None:
            return []
        questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
        items: list[QAItem] = []
        for q in questions or []:
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", "") or "").strip()
            if not question:
                continue
            items.append(
                QAItem(
                    question=question,
                    answer=str(q.get("answer", "") or "").strip(),
                    explanation=str(q.get("explanation", "") or "").strip(),
                )
            )
        return items

    # ----------------------------------------------------------------- HTTP
    def _chat(self, body: dict) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return _with_retry(lambda: self._post(url, payload, headers))

    def _post(self, url: str, payload: bytes, headers: dict) -> str:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", errors="replace")
            raise BackendError(_pretty_http_error(exc.code, body_txt, self._model)) from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"Could not reach the OpenAI-compatible server at {self._base_url}. "
                f"Is it running and the Base URL correct? Underlying error: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise BackendError(f"OpenAI-compatible request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Server returned non-JSON response: {raw[:200]}") from exc

        try:
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            # Some servers nest errors differently; surface what we can.
            err = (data or {}).get("error")
            if err:
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise BackendError(f"Server error: {msg}") from exc
            raise BackendError(f"Unexpected response shape: {str(data)[:200]}") from exc


# ---------------------------------------------------------------------- retry
def _with_retry(fn: Callable[[], T]) -> T:
    """Retry transient transport failures with exponential backoff.

    Provider CDNs reset the connection on large or bursty uploads, so the same
    request usually succeeds on a second attempt. Auth/quota errors are raised
    immediately — retrying those only wastes API calls.
    """
    delay = _INITIAL_BACKOFF_S
    last_err: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn()
        except BackendError as e:
            last_err = e
            if not _is_transient(e) or attempt == _MAX_ATTEMPTS:
                raise
        except Exception as e:
            last_err = e
            if attempt == _MAX_ATTEMPTS:
                raise
        log.info(
            "OpenAI-compatible transient error (attempt %d/%d): %s — retrying in %.1fs",
            attempt,
            _MAX_ATTEMPTS,
            last_err,
            delay,
        )
        time.sleep(delay)
        delay *= 2

    assert last_err is not None
    raise last_err


def _is_transient(err: BackendError) -> bool:
    """True for errors that a retry can plausibly fix."""
    text = str(err).lower()
    if "authentication failed" in text or "http 401" in text or "http 403" in text:
        return False
    if "http 400" in text:
        return False
    if "http 429" in text:
        return True
    return any(
        marker in text
        for marker in (
            "connection reset",
            "closed connection",
            "broken pipe",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "http 550",
        )
    )


# ---------------------------------------------------------------- helpers
def _parse_json_object(raw: str) -> dict | list | None:
    """Parse a JSON object out of a model reply, tolerating markdown fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _pretty_http_error(code: int, body: str, model: str) -> str:
    low = body.lower()
    if code == 401:
        return "Authentication failed (HTTP 401). Check your API key in Settings."
    if code == 404 and "model" in low:
        return f"Model '{model}' not found on this server (HTTP 404). Check the model name."
    if code == 400 and ("image" in low or "vision" in low or "multimodal" in low):
        return f"Model '{model}' may not support images (HTTP 400). Pick a vision-capable model."
    if code == 429:
        return "Rate limited (HTTP 429). Slow down or check your quota."
    if code in (502, 503, 504):
        return "Server temporarily unavailable. Try again in a moment."
    snippet = body.strip().replace("\n", " ")
    if len(snippet) > 200:
        snippet = snippet[:197] + "…"
    return f"HTTP {code}: {snippet}"
