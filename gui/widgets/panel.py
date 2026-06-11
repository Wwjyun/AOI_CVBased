from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.theme import PAD_PANEL

# ============================================================
# AOI Console — Panel container (port of `.panel` in ui.css)
# ============================================================


class Panel(QFrame):
    """Card with optional header (title + actions) and a body area."""

    def __init__(self, title: str = "", actions: QWidget | None = None, flush: bool = False, parent=None):
        super().__init__(parent)
        self.setProperty("role", "panel")

        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(0)

        self._title_label: QLabel | None = None
        self._header_layout: QHBoxLayout | None = None
        self._actions_widget: QWidget | None = None

        if title or actions is not None:
            header = QFrame()
            header.setProperty("role", "panel-header")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(PAD_PANEL, 11, PAD_PANEL, 11)

            title_label = QLabel(title)
            title_label.setProperty("role", "panel-title")
            header_layout.addWidget(title_label)
            header_layout.addStretch(1)

            if actions is not None:
                header_layout.addWidget(actions)

            self._title_label = title_label
            self._header_layout = header_layout
            self._actions_widget = actions

            self._outer.addWidget(header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        if flush:
            self.body_layout.setContentsMargins(0, 0, 0, 0)
        else:
            self.body_layout.setContentsMargins(PAD_PANEL, PAD_PANEL, PAD_PANEL, PAD_PANEL)
        self.body_layout.setSpacing(10)

        self._outer.addWidget(self.body, 1)

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self.body_layout.addWidget(widget, stretch)

    def add_layout(self, layout) -> None:
        self.body_layout.addLayout(layout)

    def set_title(self, text: str) -> None:
        if self._title_label is not None:
            self._title_label.setText(text)

    def set_actions(self, widget: QWidget) -> None:
        if self._header_layout is None:
            return
        if self._actions_widget is not None:
            self._header_layout.removeWidget(self._actions_widget)
            self._actions_widget.deleteLater()
        self._header_layout.addWidget(widget)
        self._actions_widget = widget

    def clear_body(self) -> None:
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                _clear_layout(layout)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        sub_layout = item.layout()
        if sub_layout is not None:
            _clear_layout(sub_layout)
