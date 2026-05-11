"""ロギング設定モジュール.

プロジェクト共通のロガーを提供する。ファイル出力のみ、
RotatingFileHandlerで世代管理する。
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# プロジェクトルート直下の logs/ にログを集約する
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _PROJECT_ROOT / "logs"
_DEFAULT_LOG_FILE = "app.log"

# ログのフォーマット: 2026-05-10 10:30:00 | INFO | name | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ローテーション設定: 5MBごと、3世代まで保持
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = _DEFAULT_LOG_FILE,
) -> logging.Logger:
    """名前付きロガーを取得する.

    Args:
        name: ロガー名（通常は ``__name__`` を渡す）。
        level: ログレベル。デフォルトは ``logging.INFO``。
        log_file: 出力先ファイル名。``logs/`` 配下に作成される。
            モジュールごとに別ファイルへ分けたい場合に指定する。

    Returns:
        ファイル出力ハンドラを設定済みのロガー。
    """
    logger = logging.getLogger(name)

    # 二重設定を防ぐため、既にハンドラが付いていたら再設定しない
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / log_file

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    return logger
