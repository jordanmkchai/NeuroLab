"""Local mouse-wheel routing for scrollable Tkinter regions."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable
from typing import Any


def _wheel_units(event: Any) -> int:
    """Translate Windows/macOS and X11 wheel events into scroll units."""
    button = getattr(event, "num", None)
    if button == 4:
        return -3
    if button == 5:
        return 3

    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    steps = max(1, abs(delta) // 120)
    return -steps if delta > 0 else steps


class MousewheelBinding:
    """Callable local wheel router with incremental binding and cleanup."""

    _SEQUENCES = ("<MouseWheel>", "<Button-4>", "<Button-5>")
    def __init__(
        self,
        scrollable: Any,
        *regions: Any,
        enabled: Callable[[], bool] | None = None,
        excluded_widget_classes: Iterable[str] | None = None,
    ) -> None:
        self._scrollable_ref = weakref.ref(scrollable)
        self._enabled = enabled
        self._excluded_widget_classes = frozenset(excluded_widget_classes or ())
        self._targets: dict[int, tuple[weakref.ReferenceType[Any], list[tuple[str, Any]]]] = {}
        self._cancelled = False
        self.add_regions(scrollable, *regions)

    def __call__(self, event: Any) -> str | None:
        if self._cancelled:
            return None
        scrollable = self._scrollable_ref()
        if scrollable is None:
            self.cancel()
            return None
        if self._enabled is not None and not self._enabled():
            return None
        units = _wheel_units(event)
        if units == 0:
            return None
        try:
            scrollable.yview_scroll(units, "units")
        except Exception:
            return None
        return "break"

    def add_regions(self, *regions: Any) -> None:
        """Add local wheel regions without duplicating existing bindings."""
        if self._cancelled:
            return
        for target in regions:
            if target is None or id(target) in self._targets:
                continue
            scrollable = self._scrollable_ref()
            if target is not scrollable:
                try:
                    widget_class = target.winfo_class()
                except Exception:
                    widget_class = ""
                if widget_class in self._excluded_widget_classes:
                    continue
            try:
                target_ref = weakref.ref(target)
            except TypeError:
                continue
            bindings: list[tuple[str, Any]] = []
            for sequence in self._SEQUENCES:
                try:
                    funcid = target.bind(sequence, self, add="+")
                except Exception:
                    continue
                bindings.append((sequence, funcid))
            self._targets[id(target)] = (target_ref, bindings)

    def cancel(self) -> None:
        """Remove this router's local bindings and release widget callbacks."""
        if self._cancelled:
            return
        self._cancelled = True
        for target_ref, bindings in self._targets.values():
            target = target_ref()
            if target is None:
                continue
            for sequence, funcid in bindings:
                if not funcid:
                    continue
                try:
                    target.unbind(sequence, funcid)
                except Exception:
                    pass
        self._targets.clear()
        self._enabled = None
        self._scrollable_ref = lambda: None

    unbind = cancel


def bind_vertical_mousewheel(
    scrollable: Any,
    *regions: Any,
    enabled: Callable[[], bool] | None = None,
    excluded_widget_classes: Iterable[str] | None = None,
) -> MousewheelBinding:
    """Bind wheel scrolling only to local regions, never globally."""
    return MousewheelBinding(
        scrollable,
        *regions,
        enabled=enabled,
        excluded_widget_classes=excluded_widget_classes,
    )


__all__ = ["MousewheelBinding", "bind_vertical_mousewheel"]
