"""Shared types and base class for LLM backends."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """Outcome of running vision on a screenshot.

    `questions` is the new field — backends may extract multiple questions from
    one screenshot (e.g. a practice page with several MCQs). For backward-
    compatibility, `question` mirrors `questions[0]` when present.
    """

    has_question: bool
    question: str = ""
    questions: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        # Normalise — if only `question` was set, populate the list; if only
        # the list was set, populate the singular field.
        if self.questions and not self.question:
            self.question = self.questions[0]
        elif self.question and not self.questions:
            self.questions = [self.question]


@dataclass
class AnswerResult:
    answer: str
    explanation: str


@dataclass
class QAItem:
    """One extracted question + its answer. Used by the OCR-first pipeline and
    the single-call cloud reader (`answer_screen`)."""

    question: str
    answer: str = ""
    explanation: str = ""
    requires_visual_context: bool = False
    confidence: float = 0.0


class BackendError(Exception):
    """Raised by backends when an API call fails. The string form is shown to
    the user, so keep it short and human-readable."""


# -------------------------------------------------------------------- prompts
# Detection now requests up to 5 questions. We cap the list to keep responses
# small — most screenshots have one focused question; practice-PDF pages might
# have 3–5.
DETECT_SYSTEM = """\
You are a vision assistant that inspects screenshots to find quiz, exam, \
homework, or test-style questions a user might want help answering.

Respond ONLY with a single JSON object, no prose, no markdown fences. Schema:

{
  "has_question": boolean,
  "questions": [string, ...],  // up to 5 extracted questions, in order of \
prominence on the page. Empty array if has_question is false.
  "notes": string             // optional short context, e.g. "multiple choice", \
"math", "code". Empty if none.
}

Rules:
- has_question is true ONLY if the screenshot clearly contains at least one \
question the user is meant to answer (quiz, test, exercise, flashcard, MCQ, \
fill-in-the-blank, short answer, coding problem, math problem, etc).
- General UI text, chat messages, casual writing, news, social media, code \
editors without an explicit problem statement, etc. are NOT questions.
- For each extracted question: preserve the wording exactly. Include any \
answer choices verbatim, each on its own line, prefixed with their label \
(A., B., 1), etc.).
- If multiple questions are visible, include each as a separate string in \
the questions array. Maximum 5. List them in the order they appear (most \
prominent / top first).
- Output only the JSON object. No commentary."""


ANSWER_SYSTEM = """\
You are a careful expert tutor. The user will give you a single quiz or exam \
question. Reason through it step by step internally, then give a clean answer.

Output format — plain text, exactly two sections with these literal headers:

ANSWER:
<the answer itself, as concise as the question allows. For multiple choice, \
state the letter AND the full option text, e.g. "B. Mitochondria". For numeric, \
give the number with units. For short answer, give the answer phrase.>

EXPLANATION:
<a clear, friendly explanation of why this is correct. 2-6 sentences. \
If relevant, briefly say why the other options are wrong.>

Do not use markdown headers, bullets, or bold. Just the two labelled sections."""


# ------------------------------------------------------------------ image upload
# Provider CDNs reject uploads on their own schedule: the edge in front of
# DeepSeek (TencentEdgeOne) intermittently answers image requests with HTTP 550,
# and the odds climb with the size of the body. An earlier 20 KB cap dodged
# those rejections but squeezed dense screens down to ~600px, where diagrams
# and formulas stop being legible. Re-measured on 2026-11-10: 84 KB and 135 KB
# bodies both passed 8/8, so the cap now sits at 150 KB and the retry ladder in
# the OpenAI backend absorbs the occasional 550.
#
# Only the wire payload is compressed — the local OCR path still reads the
# original PNG, so on-device text recognition keeps full fidelity.
MAX_UPLOAD_BYTES = 150_000

# Longest edge of the uploaded image, and the quality ladder used to squeeze a
# screenshot under MAX_UPLOAD_BYTES. 1568px keeps diagrams, formulas and small
# labels legible for the vision model; the earlier 20 KB budget forced dense
# screens down to ~600px, which made figures unreadable.
_UPLOAD_MAX_DIM = 1568
_UPLOAD_QUALITIES = (80, 70, 60, 50, 40, 30)
_UPLOAD_MAX_DIMS = (1568, 1280, 1024, 900, 800, 700, 600, 500, 400)


def sniff_image_mime(data: bytes) -> str:
    """Best-effort MIME type of an encoded image, from its magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF8"):
        return "image/gif"
    return "image/png"


def encode_image_for_upload(
    image_bytes: bytes, max_bytes: int | None = None
) -> tuple[str, bytes]:
    """Return (mime_type, image_bytes) small enough for a provider request.

    Images already under `max_bytes` are passed through untouched. Larger ones
    are re-encoded as JPEG, first lowering quality and then shrinking the
    image, until the result fits.
    """
    if max_bytes is None:
        max_bytes = MAX_UPLOAD_BYTES
    if len(image_bytes) <= max_bytes:
        return sniff_image_mime(image_bytes), image_bytes

    import io

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        # Not a decodable image — send it as-is rather than dropping the payload.
        return sniff_image_mime(image_bytes), image_bytes

    w, h = img.size
    longest = max(w, h)

    encoded = b""
    # Quality first, then size: dropping quality costs legibility, shrinking
    # the image costs detail, so keep as much of the former as possible.
    for max_dim in _UPLOAD_MAX_DIMS:
        if longest > max_dim:
            scale = max_dim / longest
            scaled = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
            )
        else:
            scaled = img
        for quality in _UPLOAD_QUALITIES:
            buf = io.BytesIO()
            scaled.save(buf, format="JPEG", quality=quality, optimize=True)
            encoded = buf.getvalue()
            if len(encoded) <= max_bytes:
                return "image/jpeg", encoded
    return "image/jpeg", encoded


# ------------------------------------------------------------------- base class
class Backend(ABC):
    """Provider-agnostic interface for the rest of the app to use."""

    name: str = "base"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("API key is required")
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @abstractmethod
    def detect_and_extract_question(self, png_bytes: bytes) -> DetectionResult:
        """Vision: decide if questions are on screen and extract them."""

    @abstractmethod
    def answer_question(self, question: str, context: str | None = None) -> AnswerResult:
        """Text: reason through and answer a single question.

        `context`, when provided, is prior conversation text the model should
        use as background (e.g. "Previous Q: …\nPrevious A: …"). Backends
        should prepend it to the user turn rather than treat it as the system
        prompt — keeps the ANSWER format contract intact.
        """

    def warmup(self) -> None:
        """Optional: preload the model so the first real call isn't cold.

        Default is a no-op (cloud backends need no warmup). Local backends
        override this to load weights + vision encoder into memory ahead of
        the user's first capture. Must never raise.
        """
        return

    def answer_questions_with_image(
        self, questions: list[str], png_bytes: bytes
    ) -> list[AnswerResult]:
        """Vision: answer already-extracted questions using the screenshot.

        Used for visually-dependent questions (charts, diagrams, geometry) where
        OCR text alone can't answer. One call answers all `questions`, in order.
        Default: unsupported.
        """
        raise NotImplementedError

    def answer_screen(self, ocr_text: str, png_bytes: bytes) -> list[QAItem]:
        """One-call screen reader: given OCR text + the screenshot, extract every
        question and answer each in a SINGLE request. Lets cloud backends handle
        a whole capture in one API call (1 request instead of detect + N answers),
        using the image for charts/figures and the OCR text for exact wording.
        Default: unsupported.
        """
        raise NotImplementedError


def build_followup_prompt(context: str, question: str) -> str:
    """Compose the user message for a follow-up question."""
    return (
        "Earlier in this conversation:\n"
        f"{context.strip()}\n\n"
        "Follow-up question:\n"
        f"{question.strip()}"
    )


# ============================================================ shared parsers
def parse_detection_json(raw: str) -> DetectionResult:
    """Forgiving extractor — handles clean JSON, markdown-fenced JSON, JSON in
    prose, and the older single-question schema."""
    if not raw:
        return DetectionResult(has_question=False, notes="empty response")

    candidate = _strip_fences(raw)
    if not candidate.lstrip().startswith("{"):
        m = re.search(r"\{[\s\S]*\}", candidate)
        if m:
            candidate = m.group(0)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return DetectionResult(has_question=False, notes="parse error")

    if not isinstance(data, dict):
        # E.g. the API returned a list or a bare scalar — treat as no question.
        return DetectionResult(has_question=False, notes="unexpected JSON shape")

    has = bool(data.get("has_question", False))
    notes = str(data.get("notes", "") or "").strip()

    # Prefer the new `questions` array; fall back to legacy `question` string.
    raw_list = data.get("questions")
    questions: list[str] = []
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    questions.append(t)
    elif data.get("question"):
        t = str(data["question"]).strip()
        if t:
            questions.append(t)

    # Cap at 5 to match the prompt contract.
    questions = questions[:5]

    if has and not questions:
        # Inconsistent — treat as no question.
        has = False

    return DetectionResult(has_question=has, questions=questions, notes=notes)


def parse_answer_text(raw: str) -> AnswerResult:
    """Split the ANSWER/EXPLANATION sections. Tolerant of small format drift."""
    if not raw:
        return AnswerResult(answer="(no response)", explanation="")

    m = re.search(
        r"ANSWER\s*:\s*(.+?)\n\s*EXPLANATION\s*:\s*(.+)$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return AnswerResult(answer=m.group(1).strip(), explanation=m.group(2).strip())
    return AnswerResult(answer=raw.strip(), explanation="")


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()
