"""Tests for the overlay's multi-question paginator state machine.

These use the offscreen Qt platform (set in conftest) — no real window pops up.
"""

from __future__ import annotations


def test_paginator_starts_at_first(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1", "Q2", "Q3"])
    assert ov.total_entries() == 3
    assert ov.current_index() == 0


def test_set_answer_updates_specific_slot(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1", "Q2"])
    ov.set_answer_for(0, "A1", "explain")
    ov.set_answer_for(1, "A2", "explain2")
    assert ov._entries[0].answer == "A1"
    assert ov._entries[1].answer == "A2"


def test_set_answer_out_of_range_is_noop(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1"])
    ov.set_answer_for(99, "x", "y")  # should not crash
    assert ov._entries[0].answer is None


def test_show_single_answer_compat(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_single_answer("Q", "A", "explain")
    assert ov.total_entries() == 1
    assert ov._entries[0].answer == "A"


def test_show_questions_with_empty_list(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    # Empty list should fall back to the "no question" state, not crash.
    ov.show_questions([])
    assert ov.total_entries() == 0


def test_filters_whitespace_only_questions(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1", "  ", "", "Q2"])
    assert ov.total_entries() == 2


def test_set_error_for_specific_slot(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1", "Q2"])
    ov.set_error_for(1, "rate limited")
    assert ov._entries[1].answer is not None
    assert "rate limited" in ov._entries[1].answer


# --------------------------------------------------------------- transparency
def test_background_opacity_controls_the_panel_alpha(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(background_opacity=0.15)
    line = next(
        ln for ln in ov._qss().splitlines() if "background-color" in ln
    )
    # 0.15 * 235 rounds to 35.
    assert "rgba(20, 22, 28, 35)" in line


def test_fully_transparent_background_is_allowed(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(background_opacity=0.0)
    assert ov._background_opacity == 0.0
    line = next(
        ln for ln in ov._qss().splitlines() if "background-color" in ln
    )
    assert "rgba(20, 22, 28, 0)" in line


def test_set_background_opacity_updates_the_style(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(background_opacity=0.85)
    ov.set_background_opacity(0.4)
    assert ov._background_opacity == 0.4
    line = next(
        ln for ln in ov._root.styleSheet().splitlines() if "background-color" in ln
    )
    assert "rgba(20, 22, 28, 94)" in line  # 0.4 * 235


def test_background_opacity_is_clamped(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(background_opacity=5.0)
    assert ov._background_opacity == 1.0
    ov.set_background_opacity(-1.0)
    assert ov._background_opacity == 0.0


def test_window_opacity_floor_is_low_enough_for_transparency(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(opacity=0.0)
    assert ov._opacity == 0.1


def test_zero_background_leaves_no_frame(qapp):
    """At 0 the panel must vanish completely: no border, no drop shadow."""
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(background_opacity=0.0)
    border = next(ln for ln in ov._qss().splitlines() if "border:" in ln)
    assert "transparent" in border
    assert ov._root_shadow.isEnabled() is False


def test_partial_background_keeps_a_faint_border(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow(background_opacity=0.5)
    border = next(ln for ln in ov._qss().splitlines() if "border:" in ln)
    assert "rgba(255, 255, 255, 20)" in border
    assert ov._root_shadow.isEnabled() is True


def test_overlay_stays_on_screen_when_the_app_is_inactive(qapp):
    """Qt.Tool is an NSPanel on macOS and hides as soon as another app is
    focused — typing anywhere else made the answer box vanish."""
    import sys

    from PySide6.QtCore import Qt

    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    assert ov.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    if sys.platform == "darwin":
        assert ov.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)


def test_answer_is_never_auto_hidden(qapp):
    """A delivered answer must stay until dismissed — no timer."""
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1"])
    ov.set_answer_for(0, "A1", "because")
    assert ov._auto_hide_timer.isActive() is False

    ov.set_error_for(0, "boom")
    assert ov._auto_hide_timer.isActive() is False


# --------------------------------------------------------------- panel height
def test_panel_grows_to_fit_the_explanation(qapp):
    """The panel must be tall enough for every line of the answer.

    QLayout ignores hidden children, so a size hint measured while the panel
    is still hidden comes back as zero and the explanation gets clipped.
    """
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["下列几何体的体积公式中，正确的是："])
    ov.set_answer_for(0, "B. 圆柱 V = πr²h", "圆柱的体积等于底面积乘以高。")

    content_h = ov._content_height()
    assert content_h > 60  # three lines of text plus spacing

    # The window has to be at least as tall as the content plus the chrome.
    assert ov.height() >= content_h


def test_rendered_labels_are_visible(qapp):
    """Freshly rendered content must not stay hidden behind the layout."""
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1"])
    ov.set_answer_for(0, "A1", "because")

    for container in ov._entry_containers:
        assert container.isHidden() is False
        for i in range(container.layout().count()):
            w = container.layout().itemAt(i).widget()
            if w is not None:
                assert w.isHidden() is False


def test_rerender_replaces_old_content(qapp):
    from quizai.overlay import OverlayWindow

    ov = OverlayWindow()
    ov.show_questions(["Q1"])
    ov.set_answer_for(0, "A1", "because")
    first = list(ov._entry_containers)

    ov.set_answer_for(0, "A2", "otherwise")
    assert len(ov._entry_containers) == 1
    assert ov._entry_containers[0] is not first[0]
