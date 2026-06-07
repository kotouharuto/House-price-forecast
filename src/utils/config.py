"""設定値を管理するモジュール.

``configs/model_params.yaml`` などの YAML 設定を読み込み、モデル学習側の
コードに渡すための薄いラッパを提供する。
"""

from pathlib import Path
from typing import Any

import yaml

# プロジェクトルート基準の絶対パス（呼び出し場所に依存しないようにする）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PARAMS_PATH = _PROJECT_ROOT / "configs" / "model_params.yaml"

# モデル設定 yaml に必須のトップレベルセクション
_REQUIRED_SECTIONS: tuple[str, ...] = ("lgbm", "split")


def load_model_params(
    file_path: str | Path = _DEFAULT_MODEL_PARAMS_PATH,
) -> dict[str, Any]:
    """``model_params.yaml`` を読み込み、必須セクションの存在を検証する.

    Args:
        file_path: 設定 YAML のパス。デフォルトは ``configs/model_params.yaml``。

    Returns:
        ``{"lgbm": {...}, "split": {...}}`` の辞書。

    Raises:
        FileNotFoundError: 設定ファイルが存在しない場合。
        KeyError: 必須セクション（``lgbm`` / ``split``）が欠落している場合。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"モデル設定が見つかりません: {path}")

    with path.open(encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f) or {}

    missing = [key for key in _REQUIRED_SECTIONS if key not in params]
    if missing:
        raise KeyError(f"model_params.yaml に必須セクションがありません: {missing} (path={path})")

    return params
