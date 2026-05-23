"""設定値を管理するモジュール."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 設定ファイルの配置先（プロジェクトルート基準の絶対パスで固定）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PARAMS_PATH = _PROJECT_ROOT / "configs" / "model_params.yaml"


def load_model_params(
    section: str = "lgbm",
    params_path: Path = _MODEL_PARAMS_PATH,
) -> dict[str, Any]:
    """YAMLファイルからモデルのハイパーパラメータを読み込む.

    設定値（ハイパーパラメータ）をコードから分離し、``configs/model_params.yaml``
    で一元管理するためのローダ。指定セクションの内容をそのまま辞書で返すため、
    ``LGBMRegressor(**load_model_params("lgbm"))`` のように展開して利用できる。

    Args:
        section: 読み込むパラメータのセクション名（例: ``"lgbm"``）。
        params_path: パラメータYAMLファイルのパス。デフォルトは
            ``configs/model_params.yaml``。

    Returns:
        ハイパーパラメータ名をキーとする辞書。

    Raises:
        FileNotFoundError: 指定したYAMLファイルが存在しない場合。
        TypeError: YAML全体またはセクションがマッピング（辞書）でない場合。
        KeyError: 指定したセクションがYAML内に存在しない場合。
    """
    if not params_path.exists():
        logger.error(f"モデルパラメータファイルが見つかりません: {params_path}")
        raise FileNotFoundError(f"モデルパラメータファイルが見つかりません: {params_path}")

    with params_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # YAML全体が辞書（マッピング）であることを保証する
    if not isinstance(config, dict):
        logger.error(f"パラメータファイルの形式が不正です（辞書を想定）: {params_path}")
        raise TypeError(f"パラメータファイルの形式が不正です（辞書を想定）: {params_path}")

    if section not in config:
        logger.error(f"セクション '{section}' が {params_path} に存在しません")
        raise KeyError(f"セクション '{section}' が {params_path} に存在しません")

    params = config[section]

    # セクション直下もマッピングであることを保証する
    if not isinstance(params, dict):
        logger.error(f"セクション '{section}' はマッピングである必要があります: {params_path}")
        raise TypeError(f"セクション '{section}' はマッピングである必要があります: {params_path}")

    logger.info(f"モデルパラメータを読み込みました: section={section}, keys={list(params.keys())}")
    return params
