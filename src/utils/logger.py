"""ロギング設定モジュール.

プロジェクト共通のロガーを提供する。ファイル出力のみ、
RotatingFileHandlerで世代管理する。
"""

from __future__ import annotations

import logging
import sys
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


def _resolve_log_filename(name: str) -> str:
    """ロガー名から出力ファイル名を導出する.

    例:
        ``src.preprocessing.clean`` -> ``clean.log``
        ``src.modeling.train``      -> ``train.log``
        ``__main__`` (python train.py 実行時) -> 実行スクリプト名から ``train.log``
        導出不能な場合                -> ``app.log`` (デフォルトへフォールバック)

    Args:
        name: ロガー名（``__name__`` で渡される）。

    Returns:
        ``logs/`` 配下に作成するログファイル名。
    """
    # ドット区切りの最後の要素をモジュール名として採用
    module_name = name.rsplit(".", 1)[-1]

    # __main__（python xxx.py で直接実行）の場合は実行スクリプト名から導出する
    # （例: python src/modeling/train.py -> "train.log"）
    if module_name == "__main__":
        script_stem = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ""
        if script_stem and script_stem != "__main__":
            return f"{script_stem}.log"
        return _DEFAULT_LOG_FILE

    # 空文字など特殊ケースはデフォルトへ集約
    if not module_name:
        return _DEFAULT_LOG_FILE

    return f"{module_name}.log"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """名前付きロガーを取得する.

    出力先のログファイルは ``name`` から自動的に導出される。
    例: ``get_logger("src.preprocessing.clean")`` -> ``logs/clean.log``

    Args:
        name: ロガー名（通常は ``__name__`` を渡す）。
        level: ログレベル。デフォルトは ``logging.INFO``。
        log_file: 出力先ファイル名を明示指定したい場合に渡す。
            ``None`` の場合は ``name`` から自動導出される。

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

    # 明示指定が無ければ name から自動導出
    resolved_log_file = log_file if log_file is not None else _resolve_log_filename(name)
    log_path = _LOG_DIR / resolved_log_file

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)

    return logger
