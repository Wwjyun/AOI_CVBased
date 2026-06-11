from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.theme import COLORS
from gui.widgets.common import IconButton

DRAWER_WIDTH = 380

# ============================================================
# AOI Console — right-side sliding settings drawer
# ============================================================


class Drawer(QWidget):
    """Overlay covering the parent widget with a right-side sliding panel."""

    closed = Signal()

    def __init__(self, title: str, parent: QWidget):
        super().__init__(parent)
        self.setVisible(False)
        self.setAutoFillBackground(False)
        parent.installEventFilter(self)

        self._panel = QFrame(self)
        self._panel.setFixedWidth(DRAWER_WIDTH)
        self._panel.setStyleSheet(
            f"QFrame {{ background: {COLORS['surface']}; border-left: 1px solid {COLORS['border']}; }}"
        )

        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        header = QFrame()
        header.setProperty("role", "panel-header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        title_label = QLabel(title)
        title_label.setProperty("role", "panel-title")
        title_label.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        close_button = IconButton("x", "關閉")
        close_button.clicked.connect(self.close_drawer)
        header_layout.addWidget(close_button)
        panel_layout.addWidget(header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(16, 16, 16, 16)
        self.body_layout.setSpacing(14)
        self.body_layout.addStretch(1)
        panel_layout.addWidget(self.body, 1)

        self._animation = QPropertyAnimation(self._panel, b"pos")
        self._animation.setDuration(160)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.insertWidget(self.body_layout.count() - 1, widget)

    def add_layout(self, layout) -> None:
        self.body_layout.insertLayout(self.body_layout.count() - 1, layout)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self._sync_geometry(animate=False)
        return super().eventFilter(watched, event)

    def _sync_geometry(self, animate: bool) -> None:
        parent_rect = self.parent().rect()
        self.setGeometry(parent_rect)
        panel_height = parent_rect.height()
        self._panel.setFixedHeight(panel_height)
        end_x = parent_rect.width() - DRAWER_WIDTH
        if not self.isVisible():
            self._panel.move(parent_rect.width(), 0)
            return
        if animate:
            self._animation.stop()
            self._animation.setStartValue(self._panel.pos())
            self._animation.setEndValue(self._panel.pos().__class__(end_x, 0))
            self._animation.start()
        else:
            self._panel.move(end_x, 0)

    def open_drawer(self) -> None:
        self.raise_()
        self.setVisible(True)
        self._sync_geometry(animate=False)
        self._panel.move(self.parent().rect().width(), 0)
        self._sync_geometry(animate=True)
        self.setFocus()

    def close_drawer(self) -> None:
        if not self.isVisible():
            return
        end_x = self.parent().rect().width()
        self._animation.stop()
        self._animation.setStartValue(self._panel.pos())
        self._animation.setEndValue(self._panel.pos().__class__(end_x, 0))
        self._animation.finished.connect(self._on_close_finished)
        self._animation.start()

    def _on_close_finished(self) -> None:
        self._animation.finished.disconnect(self._on_close_finished)
        self.setVisible(False)
        self.closed.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 70))

    def mousePressEvent(self, event) -> None:
        if not self._panel.geometry().contains(event.pos()):
            self.close_drawer()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close_drawer()
        else:
            super().keyPressEvent(event)
