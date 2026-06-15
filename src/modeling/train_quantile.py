"""分位点回帰モデル（予測区間）を学習するモジュール.

``configs/model_params.yaml`` の ``quantile`` セクションに従い、α ごとに独立した
LightGBM モデルを学習する。説明可能性ロードマップ Phase 2 の主要成果物。

- 既存の ``lgbm_model.pkl`` には影響を与えない（補助モデル扱い）。
- 学習後にテストセットでカバレッジ率（PICP）と区間幅をログ出力する。
"""

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.modeling.train import prepare_dataset  # noqa: E402
from src.utils.config import load_model_params  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

# モデル保存先（モジュール import 時に保存先ディレクトリを保証する）
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 分位点 3 つを low / med / high にマッピングするための固定名
_POSITION_NAMES: tuple[str, str, str] = ("low", "med", "high")

# 想定する分位点数（low / median / high の 3 つ）
_EXPECTED_NUM_ALPHAS = 3

logger = get_logger(__name__)


def build_quantile_params(
    lgbm_params: dict[str, Any],
    overrides: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """lgbm セクションのパラメータに分位点回帰用の上書きと α を統合する.

    Args:
        lgbm_params: ``lgbm`` セクションのパラメータ。
        overrides: ``quantile.overrides`` セクション（objective / metric 等の上書き）。
        alpha: 分位点（0 < alpha < 1）。

    Returns:
        ``LGBMRegressor`` に渡せるパラメータ辞書。``alpha`` キーを含む。
    """
    return {**lgbm_params, **overrides, "alpha": alpha}


def train_quantile_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_params: dict[str, Any],
    overrides: dict[str, Any],
    alphas: list[float],
) -> dict[float, LGBMRegressor]:
    """α ごとに独立した LightGBM 分位点回帰モデルを学習する.

    Args:
        x_train: 学習用特徴量。
        y_train: 学習用目的変数（log スケール想定）。
        lgbm_params: ``lgbm`` セクションのパラメータ。
        overrides: ``quantile.overrides`` セクション。
        alphas: 学習する分位点のリスト。

    Returns:
        ``{alpha: 学習済みモデル}`` の辞書。
    """
    models: dict[float, LGBMRegressor] = {}
    for alpha in alphas:
        params = build_quantile_params(lgbm_params, overrides, alpha)
        model = LGBMRegressor(**params)
        model.fit(x_train, y_train)
        models[alpha] = model
        logger.info(f"Quantile モデル学習完了: α={alpha}")
    return models


def save_quantile_models(
    models: dict[float, LGBMRegressor],
    model_dir: Path = _MODEL_DIR,
) -> dict[float, Path]:
    """学習済み分位点モデルを ``models/lgbm_quantile_{low,med,high}.pkl`` に保存する.

    α を昇順に並べ、3 つを ``low`` / ``med`` / ``high`` に割り当てる。

    Args:
        models: ``{alpha: モデル}`` の辞書。3 件である必要がある。
        model_dir: 保存先ディレクトリ。

    Returns:
        ``{alpha: 保存パス}`` の辞書。

    Raises:
        ValueError: モデル数が 3 でない場合。
    """
    sorted_alphas = sorted(models.keys())
    if len(sorted_alphas) != _EXPECTED_NUM_ALPHAS:
        raise ValueError(
            f"3 つの α が必要です（low / med / high）: got {len(sorted_alphas)} 件 {sorted_alphas}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[float, Path] = {}
    for alpha, name in zip(sorted_alphas, _POSITION_NAMES, strict=True):
        path = model_dir / f"lgbm_quantile_{name}.pkl"
        joblib.dump(models[alpha], path)
        saved[alpha] = path
        logger.info(f"モデル保存: {path} (α={alpha})")
    return saved


def evaluate_coverage(
    y_test_log: pd.Series,
    models: dict[float, LGBMRegressor],
    x_test: pd.DataFrame,
) -> dict[str, float]:
    """テストセットでカバレッジ率（PICP）と区間幅（円）を計算してログ出力する.

    Args:
        y_test_log: テスト目的変数（log スケール）。
        models: ``{alpha: モデル}`` の辞書。最小・最大の α を区間端として使う。
        x_test: テスト特徴量。

    Returns:
        ``coverage_rate`` / ``nominal_coverage`` / ``width_median_yen`` /
        ``width_mean_yen`` をキーとする辞書。
    """
    sorted_alphas = sorted(models.keys())
    lower_alpha = sorted_alphas[0]
    upper_alpha = sorted_alphas[-1]
    nominal_coverage = float(upper_alpha - lower_alpha)

    lower_pred = models[lower_alpha].predict(x_test)
    upper_pred = models[upper_alpha].predict(x_test)

    # log スケールのまま判定可能（単調変換なので不等式の向きは変わらない）
    inside = (y_test_log.to_numpy() >= lower_pred) & (y_test_log.to_numpy() <= upper_pred)
    coverage_rate = float(inside.mean())

    # 区間幅は人間が読みやすい円スケールで集計
    width_yen = np.exp(upper_pred) - np.exp(lower_pred)
    width_median = float(np.median(width_yen))
    width_mean = float(np.mean(width_yen))

    logger.info(
        f"カバレッジ率: 名目 {nominal_coverage * 100:.0f}% / "
        f"実測 {coverage_rate * 100:.2f}% "
        f"(差分 {(coverage_rate - nominal_coverage) * 100:+.2f}pp)"
    )
    logger.info(f"区間幅（円）: 中央値 {width_median:,.0f} / 平均 {width_mean:,.0f}")

    return {
        "nominal_coverage": nominal_coverage,
        "coverage_rate": coverage_rate,
        "width_median_yen": width_median,
        "width_mean_yen": width_mean,
    }


def main() -> None:
    """エントリポイント: 設定読込 → データ準備 → 学習 → 保存 → カバレッジ評価."""
    start_time = time.time()

    params = load_model_params()
    if "quantile" not in params:
        raise KeyError(
            "model_params.yaml に quantile セクションがありません。"
            "configs/model_params.yaml を確認してください。"
        )

    lgbm_params = params["lgbm"]
    split_params = params["split"]
    quantile_cfg = params["quantile"]
    alphas = list(quantile_cfg["alphas"])
    overrides = quantile_cfg.get("overrides", {})

    logger.info(f"Quantile 学習設定: alphas={alphas}, overrides={overrides}")

    x, y = prepare_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

    models = train_quantile_models(x_train, y_train, lgbm_params, overrides, alphas)
    save_quantile_models(models)
    evaluate_coverage(y_test, models, x_test)

    logger.info(f"Quantile 学習完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
