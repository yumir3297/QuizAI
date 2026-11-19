"""Floating semi-transparent always-on-top overlay window.

Renders the current answer + explanation in a scrollable list when several
questions were extracted from one screenshot. Drag with the mouse to
reposition; click ✕ to dismiss; click 📌 to pin (disables auto-hide).

Multi-question state is owned by the overlay so the rest of the app can
just feed it a list of (question, answer, explanation) tuples and let the
overlay handle navigation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from quizai.logger import get_logger

log = get_logger(__name__)


@dataclass
class QAEntry:
    """One question slot in the overlay's list.

    `answer` / `explanation` are None until that question has been answered;
    the overlay shows a "thinking" placeholder for unanswered slots.
    """

    question: str
    answer: str | None = None
    explanation: str | None = None


OVERLAY_QSS = """
#overlayRoot {
    background-color: rgba(20, 22, 28, {bg_alpha});
    border: 1px solid {border_color};
    border-radius: 14px;
}
#overlayTitle {
    color: #e8eaf0;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
#overlayQuestion {
    color: #d5d9e2;
    font-size: 12px;
    font-style: italic;
}
#overlayAnswer {
    color: #ffffff;
    font-size: 16px;
    font-weight: 600;
}
#overlayExplanation {
    color: #eef0f5;
    font-size: 13px;
}
#overlayStatus {
    color: #ffd479;
    font-size: 13px;
    font-style: italic;
}
#overlayError {
    color: #ff8a8a;
    font-size: 13px;
}
QPushButton#overlayClose, QPushButton#overlayPin {
    background: transparent;
    color: #c5c9d2;
    border: none;
    font-size: 14px;
    padding: 2px 8px;
}
QPushButton#overlayClose:hover, QPushButton#overlayPin:hover {
    color: #ffffff;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
    border: none;
}
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 60);
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


class OverlayWindow(QWidget):
    """Frameless, always-on-top, semi-transparent answer display."""

    dismissed = Signal()
    # Fired when the user navigates to a question that hasn't been answered (legacy paginator support, mostly unused now since we queue all at once).
    question_answer_requested = Signal(int)
    # Emitted (width, height) after the user drag-resizes the overlay so the
    # app can persist the new geometry. Debounced until the drag settles.
    geometry_changed = Signal(int, int)

    def __init__(
        self,
        opacity: float = 0.92,
        width: int = 480,
        max_height: int = 600,
        height: int = 0,
        background_opacity: float = 0.85,
    ):
        super().__init__()
        self._drag_offset: QPoint | None = None
        self._opacity = max(0.1, min(1.0, opacity))
        self._background_opacity = max(0.0, min(1.0, background_opacity))
        self._target_width = max(280, int(width))
        self._max_height = max_height
        # >0 means the user picked an exact height by resizing; auto-fit is off.
        self._fixed_height = max(0, int(height))
        # The last size we set programmatically — used to tell our own resizes
        # apart from user drags in resizeEvent.
        self._last_set_size = (0, 0)
        self._auto_hide_timer = QTimer(self)
        self._auto_hide_timer.setSingleShot(True)
        self._auto_hide_timer.timeout.connect(self._on_auto_hide)
        # Debounce geometry saves so a drag emits once, not per pixel.
        self._geo_save_timer = QTimer(self)
        self._geo_save_timer.setSingleShot(True)
        self._geo_save_timer.timeout.connect(self._emit_geometry)
        self._pinned = False

        # State
        self._entries: list[QAEntry] = []

        self._build_ui()
        self._apply_window_flags()
        self.setWindowOpacity(self._opacity)

    # -------------------------------------------------------------- UI build
    def _qss(self) -> str:
        """Style sheet with the panel background alpha applied.

        The alpha is what makes the answer box see-through; the text itself
        keeps its own opacity so it stays legible over a transparent panel.
        The border fades out with the background: at 0 the panel must leave no
        visible frame behind, only the halo'd text.
        """
        alpha = int(round(self._background_opacity * 235))
        border_alpha = int(round(40 * self._background_opacity))
        border = (
            f"rgba(255, 255, 255, {border_alpha})" if border_alpha else "transparent"
        )
        # Plain substitution, not str.format: the stylesheet is full of CSS
        # braces that format() would choke on.
        return (
            OVERLAY_QSS.replace("{bg_alpha}", str(alpha))
            .replace("{border_color}", border)
        )

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._root = QFrame(self)
        self._root.setObjectName("overlayRoot")
        self._root.setStyleSheet(self._qss())
        # Keeps the light text readable when the panel background is cleared.
        # Disabled at zero background opacity: a dark rectangle glow would
        # re-introduce the very frame the transparent mode is meant to remove.
        shadow = QGraphicsDropShadowEffect(self._root)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 190))
        shadow.setEnabled(self._background_opacity > 0.02)
        self._root_shadow = shadow
        self._root.setGraphicsEffect(shadow)
        outer.addWidget(self._root)

        root_layout = QVBoxLayout(self._root)
        root_layout.setContentsMargins(16, 12, 16, 14)
        root_layout.setSpacing(8)
        self._root_layout = root_layout

        # Header row.
        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("QuizAI", self._root)
        title.setObjectName("overlayTitle")
        header.addWidget(self._make_readable(title))
        header.addStretch(1)

        self._pin_btn = QPushButton("📌", self._root)
        self._pin_btn.setObjectName("overlayPin")
        self._pin_btn.setToolTip("Pin (disables auto-hide)")
        self._pin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pin_btn.setCheckable(True)
        self._pin_btn.toggled.connect(self._on_pin_toggled)
        header.addWidget(self._pin_btn)

        close_btn = QPushButton("✕", self._root)
        close_btn.setObjectName("overlayClose")
        close_btn.setToolTip("Dismiss")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.dismiss)
        header.addWidget(close_btn)
        root_layout.addLayout(header)
        self._header_layout = header

        # Scrollable body.
        self._scroll = QScrollArea(self._root)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._body = QWidget(self._scroll)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)

        self._body_layout.addStretch(1)
        self._scroll.setWidget(self._body)
        root_layout.addWidget(self._scroll, 1)

        # Bottom-right drag handle to resize the overlay (frameless window).
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self._size_grip = QSizeGrip(self._root)
        self._size_grip.setToolTip("Drag to resize")
        grip_row.addWidget(
            self._size_grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        )
        root_layout.addLayout(grip_row)
        self._grip_layout = grip_row

        # Resizable instead of fixed-width so the grip works in both axes.
        self.setMinimumWidth(280)
        self.setMinimumHeight(140)
        self._apply_size(self._target_width, self._fixed_height or 160)

    def _apply_window_flags(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        if sys.platform == "darwin":
            # Qt.Tool maps to NSPanel on macOS, and NSPanel disappears as soon
            # as the app loses focus — the answer box vanished the moment the
            # user typed in another app. This attribute keeps it on screen while
            # QuizAI is inactive.
            self.setAttribute(
                Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True
            )

    def _clear_body(self):
        while self._body_layout.count() > 1:  # keeping the stretch at the end
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w:
                # Unparent first: a still-parented child keeps painting over
                # the freshly rendered content until the deferred delete runs.
                w.setParent(None)
                w.deleteLater()

    # ------------------------------------------------------------- public API
    def _make_readable(self, label: QLabel) -> QLabel:
        """Give a text label a soft dark halo.

        With a transparent panel the labels sit directly on whatever is on
        screen, so light text can end up on a light background. The halo
        keeps the text readable without re-introducing an opaque plate.
        """
        shadow = QGraphicsDropShadowEffect(label)
        shadow.setBlurRadius(6)
        shadow.setOffset(0, 1)
        shadow.setColor(QColor(0, 0, 0, 210))
        label.setGraphicsEffect(shadow)
        return label

    def _add_row(self, layout: QVBoxLayout, widget: QWidget) -> QWidget:
        """Add a widget to a layout and make sure it actually shows up.

        A widget created without a parent is reparented by the layout, and
        that reparent leaves it explicitly hidden: it never paints, and
        QLayout skips hidden items when it computes sizes. Showing it
        explicitly clears that flag.
        """
        layout.addWidget(widget)
        widget.show()
        return widget

    def set_opacity(self, opacity: float) -> None:
        self._opacity = max(0.1, min(1.0, opacity))
        self.setWindowOpacity(self._opacity)

    def set_background_opacity(self, opacity: float) -> None:
        """Set how opaque the answer panel's background is (0 = fully clear)."""
        self._background_opacity = max(0.0, min(1.0, opacity))
        if hasattr(self, "_root"):
            self._root.setStyleSheet(self._qss())
        if hasattr(self, "_root_shadow"):
            self._root_shadow.setEnabled(self._background_opacity > 0.02)

    def set_width(self, width: int) -> None:
        self._target_width = max(280, int(width))
        self._fit_height()

    def set_max_height(self, h: int) -> None:
        self._max_height = max(200, int(h))
        self._fit_height()

    def set_fixed_height(self, h: int) -> None:
        """Set a remembered exact height (0 turns auto-fit back on)."""
        self._fixed_height = max(0, int(h))
        self._fit_height()

    def show_thinking(self, message: str = "Analyzing screenshot…") -> None:
        """Initial 'looking for question' state — no entries yet."""
        self._cancel_auto_hide()
        self._entries = []
        self._clear_body()
        lbl = QLabel(message, self._body)
        lbl.setObjectName("overlayStatus")
        self._body_layout.insertWidget(0, self._make_readable(lbl))
        self._show_positioned()

    def show_error(self, message: str) -> None:
        self._cancel_auto_hide()
        self._entries = []
        self._clear_body()
        lbl = QLabel(message, self._body)
        lbl.setObjectName("overlayError")
        lbl.setWordWrap(True)
        self._body_layout.insertWidget(0, self._make_readable(lbl))
        self._show_positioned()
        self._schedule_auto_hide(10_000)

    def show_no_question(self) -> None:
        self.show_error("No question detected on screen.")

    def show_single_answer(self, question: str, answer: str, explanation: str) -> None:
        """Backwards-compatible single-question display."""
        self.show_questions([question])
        self.set_answer_for(0, answer, explanation)

    def show_questions(self, questions: list[str]) -> None:
        """Populate the list with one or more questions. The orchestrator should call
        set_answer_for(i, ...) as each answer arrives."""
        self._cancel_auto_hide()
        self._entries = [QAEntry(question=q) for q in questions if q.strip()]
        if not self._entries:
            self.show_no_question()
            return
        self._render_all()
        self._show_positioned()

    def set_answer_for(self, index: int, answer: str, explanation: str) -> None:
        """Update one slot with its answer. Re-renders."""
        if not (0 <= index < len(self._entries)):
            return
        self._entries[index].answer = answer
        self._entries[index].explanation = explanation
        self._render_all()
        # No auto-hide: an answer stays on screen until it is dismissed with
        # ✕ or Ctrl+Shift+X. Timed hiding made the panel vanish mid-read.

    def set_error_for(self, index: int, message: str) -> None:
        """Mark a specific slot as failed. Renders the message in place."""
        if not (0 <= index < len(self._entries)):
            return
        self._entries[index].answer = f"(error: {message})"
        self._entries[index].explanation = ""
        self._render_all()

    def current_index(self) -> int:
        return 0

    def total_entries(self) -> int:
        return len(self._entries)

    def dismiss(self) -> None:
        self._cancel_auto_hide()
        self.hide()
        self.dismissed.emit()

    def _render_all(self) -> None:
        self._clear_body()
        # Per-entry containers, kept so the panel height can be measured from
        # the content itself (see _content_height).
        self._entry_containers: list[QWidget] = []

        for i, entry in enumerate(self._entries):
            # Parent it to the body up front: a parentless widget gets
            # reparented by the layout while the panel is still hidden, and
            # that reparent marks it explicitly hidden — it would then never
            # appear, and QLayout skips hidden items, so the panel's height
            # could not be measured either.
            container = QWidget(self._body)
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 12)
            layout.setSpacing(6)
            # Undo the hidden state the reparent left behind. Without this the
            # container stays invisible and the layout reports a zero height.
            container.show()

            q_preview = _truncate(entry.question, 500)
            q_lbl = QLabel(f"Q: {q_preview}")
            q_lbl.setObjectName("overlayQuestion")
            q_lbl.setWordWrap(True)
            self._add_row(layout, self._make_readable(q_lbl))

            if entry.answer is None:
                # Pending
                s_lbl = QLabel("Thinking…")
                s_lbl.setObjectName("overlayStatus")
                self._add_row(layout, self._make_readable(s_lbl))
            else:
                a_lbl = QLabel(entry.answer or "(no answer)")
                a_lbl.setObjectName("overlayAnswer")
                a_lbl.setWordWrap(True)
                self._add_row(layout, self._make_readable(a_lbl))

                if entry.explanation:
                    e_lbl = QLabel(entry.explanation)
                    e_lbl.setObjectName("overlayExplanation")
                    e_lbl.setWordWrap(True)
                    self._add_row(layout, self._make_readable(e_lbl))

            # Separator if not the last item
            if i < len(self._entries) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setStyleSheet(
                    "background-color: rgba(255, 255, 255, 20); border: none; height: 1px;"
                )
                self._add_row(layout, line)

            self._entry_containers.append(container)
            self._body_layout.insertWidget(self._body_layout.count() - 1, container)

        self._fit_height()

    # ----------------------------------------------------------- positioning
    def _show_positioned(self) -> None:
        self._fit_height()
        if not self.isVisible():
            self._center_top_right()
        self.show()
        self.raise_()

    def _content_height(self) -> int:
        """Height the answer list needs, measured from the labels themselves.

        The layout's own size hint is unusable here: QLayout ignores hidden
        children, and while the panel is hidden the freshly built containers
        still count as hidden — the hint comes back as zero and the panel ends
        up too short, clipping the explanation line. Measuring each label's
        wrapped height directly avoids that whole trap.
        """
        width = max(200, self._target_width - 34)  # minus panel padding
        total = 0
        for container in getattr(self, "_entry_containers", []):
            lay = container.layout()
            if lay is None:
                continue
            for i in range(lay.count()):
                item = lay.itemAt(i)
                w = item.widget()
                if w is None:
                    continue
                h = w.heightForWidth(width)
                if h <= 0:
                    h = w.sizeHint().height()
                total += h
            total += lay.spacing() * max(0, lay.count() - 1)
            total += lay.contentsMargins().bottom()
        return total

    def _fit_height(self) -> None:
        if self._fixed_height > 0:
            # User picked an exact height — honour it; content scrolls.
            h = max(140, self._fixed_height)
        else:
            content_h = self._content_height()
            chrome = (
                self._header_layout.sizeHint().height()
                + self._grip_layout.sizeHint().height()
                + self._root_layout.contentsMargins().top()
                + self._root_layout.contentsMargins().bottom()
                + self._root_layout.spacing() * 2
            )
            # A few pixels of slack: the scroll area ends up a touch shorter
            # than the summed hints, and without the buffer the last line of
            # the explanation would sit behind the scrollbar.
            ideal = content_h + chrome + 8
            h = max(140, min(self._max_height, ideal))
        self._apply_size(self._target_width, h)

    def _apply_size(self, w: int, h: int) -> None:
        """Resize the window, recording the intended size so resizeEvent can
        distinguish our own resizes from a user drag on the size grip."""
        self._last_set_size = (int(w), int(h))
        self.resize(int(w), int(h))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Ignore resizes we triggered ourselves (auto-fit, width/opacity apply).
        if (self.width(), self.height()) == self._last_set_size:
            return
        # A real user drag: remember the new size and switch to fixed-height.
        self._target_width = self.width()
        self._fixed_height = self.height()
        self._geo_save_timer.start(500)

    def _emit_geometry(self) -> None:
        self.geometry_changed.emit(self._target_width, self._fixed_height)

    def _center_top_right(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.right() - self.width() - 24
        y = geo.top() + 60
        self.move(max(geo.left(), x), max(geo.top(), y))

    # ------------------------------------------------------------- auto-hide
    def _schedule_auto_hide(self, ms: int) -> None:
        if self._pinned:
            return
        self._auto_hide_timer.start(ms)

    def _cancel_auto_hide(self) -> None:
        self._auto_hide_timer.stop()

    def _on_auto_hide(self) -> None:
        if self._pinned:
            return
        self.hide()

    def _on_pin_toggled(self, pinned: bool) -> None:
        self._pinned = pinned
        if pinned:
            self._cancel_auto_hide()

    # ----------------------------------------------------------------- drag
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None


def _truncate(s: str, n: int) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"
