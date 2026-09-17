from types import SimpleNamespace

import pytest

from neurolab.ui.scrolling import bind_vertical_mousewheel


class _FakeWidget:
    def __init__(self):
        self.bindings = {}
        self.unbind_calls = []

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = (callback, add)
        return f"binding-{sequence}"

    def unbind(self, sequence, funcid=None):
        self.unbind_calls.append((sequence, funcid))


class _FakeScrollable(_FakeWidget):
    def __init__(self):
        super().__init__()
        self.scroll_calls = []

    def yview_scroll(self, amount, unit):
        self.scroll_calls.append((amount, unit))


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (SimpleNamespace(delta=120, num=None), -1),
        (SimpleNamespace(delta=-240, num=None), 2),
        (SimpleNamespace(delta=0, num=4), -3),
        (SimpleNamespace(delta=0, num=5), 3),
    ],
)
def test_mousewheel_binding_scrolls_canvas_from_registered_box_regions(event, expected):
    scrollable = _FakeScrollable()
    box_background = _FakeWidget()
    ignored_label = _FakeWidget()

    bind_vertical_mousewheel(scrollable, box_background)

    assert set(scrollable.bindings) == {"<MouseWheel>", "<Button-4>", "<Button-5>"}
    assert set(box_background.bindings) == {"<MouseWheel>", "<Button-4>", "<Button-5>"}
    assert ignored_label.bindings == {}
    callback, add_mode = box_background.bindings["<MouseWheel>" if event.num is None else f"<Button-{event.num}>"]
    assert add_mode == "+"
    assert callback(event) == "break"
    assert scrollable.scroll_calls == [(expected, "units")]


def test_mousewheel_binding_respects_disabled_scroll_region():
    scrollable = _FakeScrollable()
    box_background = _FakeWidget()
    bind_vertical_mousewheel(scrollable, box_background, enabled=lambda: False)

    callback, _ = box_background.bindings["<MouseWheel>"]
    assert callback(SimpleNamespace(delta=-120, num=None)) is None
    assert scrollable.scroll_calls == []


def test_mousewheel_binding_can_add_descendants_and_cleanup_without_global_unbinds():
    scrollable = _FakeScrollable()
    initial = _FakeWidget()
    descendant = _FakeWidget()
    handle = bind_vertical_mousewheel(scrollable, initial)

    handle.add_regions(descendant)
    assert set(descendant.bindings) == {"<MouseWheel>", "<Button-4>", "<Button-5>"}

    callback, _ = descendant.bindings["<MouseWheel>"]
    handle.cancel()
    assert callback(SimpleNamespace(delta=-120, num=None)) is None
    assert scrollable.scroll_calls == []
    assert descendant.unbind_calls == [
        ("<MouseWheel>", "binding-<MouseWheel>"),
        ("<Button-4>", "binding-<Button-4>"),
        ("<Button-5>", "binding-<Button-5>"),
    ]

