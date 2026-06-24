from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class AOILogConfig:
    log_dir: Path = Path("outputs") / "logs"
    level: str = "INFO"
    app_name: str = "aoi"
    max_bytes: int = 2_000_000
    backup_count: int = 5
    console: bool = True

    @classmethod
    def from_env(cls, log_dir: Path | None = None, level: str | None = None) -> "AOILogConfig":
        env_dir = os.getenv("AOI_LOG_DIR")
        env_level = os.getenv("AOI_LOG_LEVEL")
        return cls(
            log_dir=Path(log_dir or env_dir or cls.log_dir),
            level=str(level or env_level or cls.level).upper(),
        )


class AOILogManager:
    """OOP facade around Python logging for the AOI application."""

    _instance: "AOILogManager | None" = None
    _instance_lock = Lock()

    def __init__(self) -> None:
        self._configured = False
        self._config = AOILogConfig()
        self._lock = Lock()

    @classmethod
    def instance(cls) -> "AOILogManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def configure(self, config: AOILogConfig | None = None, force: bool = False) -> None:
        with self._lock:
            if self._configured and not force:
                return

            self._config = config or AOILogConfig.from_env()
            log_level = self._resolve_level(self._config.level)
            root_logger = logging.getLogger(self._config.app_name)
            root_logger.setLevel(log_level)
            root_logger.propagate = False

            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
                handler.close()

            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            self._config.log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                self._config.log_dir / f"{self._config.app_name}.log",
                maxBytes=self._config.max_bytes,
                backupCount=self._config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            if self._config.console:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)

            self._configured = True
            root_logger.info("Logging configured: dir=%s level=%s", self._config.log_dir, self._config.level)

    def get_logger(self, name: str | None = None) -> logging.Logger:
        if not self._configured:
            self.configure()
        logger_name = self._config.app_name if not name else f"{self._config.app_name}.{name}"
        return logging.getLogger(logger_name)

    @property
    def config(self) -> AOILogConfig:
        return self._config

    @staticmethod
    def _resolve_level(level: str) -> int:
        return getattr(logging, level.upper(), logging.INFO)


class LogMixin:
    @property
    def logger(self) -> logging.Logger:
        return AOILogManager.instance().get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def configure_logging(log_dir: Path | None = None, level: str | None = None, force: bool = False) -> None:
    AOILogManager.instance().configure(AOILogConfig.from_env(log_dir=log_dir, level=level), force=force)


def get_logger(name: str | None = None) -> logging.Logger:
    return AOILogManager.instance().get_logger(name)
