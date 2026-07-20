from __future__ import annotations

import hmac
from collections.abc import Mapping

from PySide6.QtWidgets import QInputDialog, QLineEdit, QWidget


MODE_LABELS = {
    "op": "OP 模式",
    "eng": "工程模式",
    "admin": "管理模式",
}

DEFAULT_MODE_PASSWORDS = {
    "eng": "1234",
    "admin": "5678",
}


class PermissionManager:
    """Owns GUI mode authorization independently from the window widgets."""

    def __init__(self, passwords: Mapping[str, str] | None = None):
        configured = dict(DEFAULT_MODE_PASSWORDS if passwords is None else passwords)
        if set(configured) != {"eng", "admin"}:
            raise ValueError("passwords must define exactly the eng and admin modes")
        self._passwords = {
            mode: str(password).encode("utf-8") for mode, password in configured.items()
        }
        self._current_mode = "op"

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def switch_mode(self, mode: str, password: str = "") -> bool:
        if mode not in MODE_LABELS:
            raise ValueError(f"unknown GUI mode: {mode}")
        if mode == "op":
            self._current_mode = mode
            return True
        if not hmac.compare_digest(str(password).encode("utf-8"), self._passwords[mode]):
            return False
        self._current_mode = mode
        return True


class ModePasswordPrompt:
    """Qt password prompt kept separate from authorization policy."""

    def request_password(self, parent: QWidget, mode: str) -> tuple[str, bool]:
        return QInputDialog.getText(
            parent,
            "權限驗證",
            f"請輸入{MODE_LABELS[mode]}密碼：",
            QLineEdit.EchoMode.Password,
        )
