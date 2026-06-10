from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QLineEdit, QScrollArea, QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget


class DetectorParamPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QLabel("Detector Parameters")
        self.enabled = QCheckBox("Enabled")
        self.enabled.setEnabled(False)
        self.form_container = QWidget()
        self.form = QFormLayout(self.form_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form_container)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.enabled)
        layout.addWidget(scroll, 1)

    def show_detector(self, detector_id: str, config: dict) -> None:
        self.title.setText(f"Detector {detector_id}")
        self.enabled.setChecked(bool(config.get("enabled", False)))
        self._clear_form()

        params = config.get("params", {})
        if not params:
            self.form.addRow(QLabel("No parameters"))
            return

        for key, value in params.items():
            widget = self._make_readonly_widget(value)
            self.form.addRow(key, widget)

    def clear(self) -> None:
        self.title.setText("Detector Parameters")
        self.enabled.setChecked(False)
        self._clear_form()

    def _clear_form(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)

    @staticmethod
    def _make_readonly_widget(value):
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            widget.setEnabled(False)
            return widget
        if isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-1_000_000, 1_000_000)
            widget.setValue(value)
            widget.setReadOnly(True)
            return widget
        if isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-1_000_000.0, 1_000_000.0)
            widget.setDecimals(4)
            widget.setValue(value)
            widget.setReadOnly(True)
            return widget
        widget = QLineEdit(str(value))
        widget.setReadOnly(True)
        return widget
