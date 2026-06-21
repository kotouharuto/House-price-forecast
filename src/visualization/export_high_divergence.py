"""中央値乖離度が高い物件を抽出して CSV 出力するスクリプト.

説明可能性ロードマップ Phase 3 の運用ツール。Quantile Regression（Phase 2）の
予測中央値と、類似物件の実取引中央値（Phase 3）の乖離が大きい物件を洗い出し、
人手査定の対象リストとして ``outputs/high_divergence_properties.csv`` に出力する。

実行例::

    uv run python -m src.visualization.export_high_divergence

乖離度・倍率・方向・重大度バンドの算出ロジックは ``prediction.py`` に集約し、
本モジュールはデータ読込 → テーブル生成 → フィルタ → 整形 → 保存の
オーケストレーションのみを担う。
"""

import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.feature_engineering import categorize_features  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization.prediction import (  # noqa: E402
    DIVERGENCE_COL,
    RATIO_COL,
    build_divergence_table,
    filter_high_divergence,
    load_quantile_models,
)

logger = get_logger(__name__)

# 入出力パス
_FEATURES_PATH = _PROJECT_ROOT / "data" / "processed" / "features.csv"
_MODEL_DIR = _PROJECT_ROOT / "models"
_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "high_divergence_properties.csv"

# 既定の抽出パラメータ
_DEFAULT_THRESHOLD = 0.30
_DEFAULT_N_SIMILAR = 30

# 出力に含める識別用の列（物件を特定し人手査定で参照する情報）
_ID_COLUMNS: tuple[str, ...] = (
    "市区町村コード",
    "住所",
    "種類",
    "面積（㎡）",
    "築年数",
    "最寄駅：距離（分）",
    "取引価格（総額）",
)

# 乖離指標の列（識別列の直後に並べる）
_METRIC_COLUMNS: tuple[str, ...] = (
    DIVERGENCE_COL,
    RATIO_COL,
    "signed_pct",
    "direction",
    "band",
)

# 補助情報の列（末尾に並べる）
_TRAILING_COLUMNS: tuple[str, ...] = (
    "quantile_median",
    "empirical_median",
    "has_overlap",
    "n_used",
    "flag",
)


def load_features(features_path: Path = _FEATURES_PATH) -> pd.DataFrame:
    """特徴量 CSV を読み込み、学習時と同じ category 変換を適用する.

    Args:
        features_path: ``features.csv`` のパス。

    Returns:
        category 変換済みの DataFrame。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
    """
    if not features_path.exists():
        raise FileNotFoundError(
            f"特徴量ファイルが見つかりません: {features_path}。"
            "先に前処理パイプラインを実行してください。"
        )
    df = pd.read_csv(features_path)
    return categorize_features(df)


def extract_model_features(models: dict[float, LGBMRegressor]) -> list[str]:
    """学習済みモデルが期待する特徴量列の順序を取り出す.

    Args:
        models: ``{alpha: モデル}`` の辞書。

    Returns:
        モデルの特徴量名のリスト。
    """
    model = next(iter(models.values()))
    return list(model.feature_name_)


def order_output_columns(table: pd.DataFrame, id_columns: Sequence[str]) -> pd.DataFrame:
    """出力列を「識別 → 乖離指標 → 補助」の順に並べ替える.

    Args:
        table: 乖離度テーブル。
        id_columns: 識別用の列。

    Returns:
        列順を整えた DataFrame（テーブルに存在しない列はスキップ）。
    """
    ordered = ["index", *id_columns, *_METRIC_COLUMNS, *_TRAILING_COLUMNS]
    present = [col for col in ordered if col in table.columns]
    return table[present]


def export_high_divergence(
    features_path: Path = _FEATURES_PATH,
    model_dir: Path = _MODEL_DIR,
    output_path: Path = _OUTPUT_PATH,
    threshold: float = _DEFAULT_THRESHOLD,
    n_similar: int = _DEFAULT_N_SIMILAR,
) -> pd.DataFrame:
    """高乖離物件を抽出して CSV に出力する.

    Args:
        features_path: 特徴量 CSV のパス。
        model_dir: 分位点モデルのディレクトリ。
        output_path: 出力先 CSV のパス。
        threshold: 「高乖離」と判定する中央値乖離度の下限。
        n_similar: 実証的区間に使う類似物件数。

    Returns:
        出力した高乖離物件の DataFrame（乖離度の降順）。
    """
    df = load_features(features_path)
    models = load_quantile_models(model_dir)
    model_features = extract_model_features(models)

    table = build_divergence_table(
        df,
        models,
        model_features=model_features,
        n_similar=n_similar,
        divergence_threshold=threshold,
        id_columns=_ID_COLUMNS,
    )

    high = filter_high_divergence(table, threshold=threshold)
    high = order_output_columns(high, _ID_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Excel での文字化けを防ぐため BOM 付き UTF-8 で保存
    high.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(
        f"高乖離物件を出力: {len(high)} 件 / 全 {len(table)} 件 "
        f"(閾値 {threshold * 100:.0f}%) → {output_path}"
    )
    return high


def main() -> None:
    """エントリポイント: 読込 → 乖離度算出 → フィルタ → CSV 出力."""
    start_time = time.time()
    high = export_high_divergence()

    # band / direction の内訳をログに残す（運用時の傾向把握用）
    if not high.empty:
        band_counts = high["band"].value_counts().to_dict()
        direction_counts = high["direction"].value_counts().to_dict()
        logger.info(f"重大度バンド内訳: {band_counts}")
        logger.info(f"方向内訳: {direction_counts}")

    logger.info(f"高乖離物件の出力完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
