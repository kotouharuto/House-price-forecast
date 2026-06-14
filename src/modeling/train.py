"""機械学習モデルを学習するモジュール.

``configs/model_params.yaml`` の LightGBM パラメータと分割設定を読み込み、
予測結果を log スケール・円スケールの両方で評価したうえでモデルを保存する。
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# プロジェクト内モジュール（sys.path 操作後である必要があるため E402 を許容）
from src.preprocessing.run_pipeline import run_pipeline  # noqa: E402
from src.utils.config import load_model_params  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

# 目的変数の列名。リーク防止のため特徴量から除外する。
_TARGET_COLUMN = "取引価格（総額）"
_LOG_TARGET_COLUMN = "log_取引価格"

# 特徴量から除外する非特徴量列。
# 取引年は類似物件検索などで使うために残しているが、モデル学習からは除外する
# （取引時期を学習に用いると将来のデータ分布に対する汎化性能が落ちる可能性があるため）。
_NON_FEATURE_COLUMNS: tuple[str, ...] = ("取引年",)

# モデル保存先（モジュール import 時に保存先ディレクトリを保証する）
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)
_MODEL_PATH = _MODEL_DIR / "lgbm_model.pkl"

logger = get_logger(__name__)


def evaluate_regression(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    """log 価格予測モデルを log スケール・円スケールの両方で評価する.

    Args:
        y_true_log: 正解値（log スケール）。
        y_pred_log: 予測値（log スケール）。

    Returns:
        ``r2_log`` / ``rmse_log`` / ``mae_yen`` / ``rmse_yen`` / ``mape`` /
        ``median_ape`` をキーとする評価指標の辞書。
    """
    y_true_yen = np.exp(y_true_log)
    y_pred_yen = np.exp(y_pred_log)

    abs_error_yen = np.abs(y_true_yen - y_pred_yen)
    ape = abs_error_yen / y_true_yen * 100

    return {
        "r2_log": r2_score(y_true_log, y_pred_log),
        "rmse_log": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "mae_yen": mean_absolute_error(y_true_yen, y_pred_yen),
        "rmse_yen": np.sqrt(mean_squared_error(y_true_yen, y_pred_yen)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
    }


def prepare_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """前処理パイプラインを実行し、特徴量 X と目的変数 y（log スケール）を返す.

    Returns:
        ``(X, y)`` のタプル。``y`` は log 変換済みの取引価格。
        取引価格列（生・log 両方）はリーク防止のため X から除外する。
    """
    df = run_pipeline()

    df[_TARGET_COLUMN] = pd.to_numeric(df[_TARGET_COLUMN], errors="coerce")
    # 0 以下の値は対数変換できないので除外
    df = df[df[_TARGET_COLUMN] > 0]
    df[_LOG_TARGET_COLUMN] = np.log(df[_TARGET_COLUMN])

    # 目的変数はリークになるため特徴量から除外する。
    # 非特徴量列（取引年など、分析用に残しているが学習では使わない列）も除外する。
    drop_columns = [_TARGET_COLUMN, _LOG_TARGET_COLUMN, *_NON_FEATURE_COLUMNS]
    drop_columns = [c for c in drop_columns if c in df.columns]
    x = df.drop(columns=drop_columns)
    y = df[_LOG_TARGET_COLUMN]
    return x, y


def log_metrics(metrics: dict[str, float]) -> None:
    """評価指標をロガーに出力する."""
    logger.info(f"R2 log score: {metrics['r2_log']:.3f}")  # 価格のばらつきの説明具合(%)
    logger.info(f"RMSE log score: {metrics['rmse_log']:.3f}")  # 価格の二乗平均平方根誤差
    logger.info(f"MAE yen score: {metrics['mae_yen']:.0f}")  # 円スケールでの平均絶対誤差
    logger.info(f"RMSE yen score: {metrics['rmse_yen']:.0f}")  # 円スケールでの二乗平均平方根誤差
    logger.info(f"MAPE score: {metrics['mape']:.3f}")  # 平均絶対パーセント誤差
    logger.info(f"Median APE score: {metrics['median_ape']:.3f}")  # 中央絶対パーセント誤差


def save_model(model: LGBMRegressor, model_path: Path = _MODEL_PATH) -> None:
    """学習済みモデルを joblib で保存する."""
    joblib.dump(model, model_path)
    logger.info(f"モデルを保存: {model_path}")


def main() -> None:
    """エントリポイント: 設定読込 → データ準備 → 学習 → 評価 → 保存."""
    start_time = time.time()

    # 設定の読み込み（モデルパラメータと分割設定は configs/model_params.yaml で管理）
    params = load_model_params()
    lgbm_params = params["lgbm"]
    split_params = params["split"]
    logger.info(f"モデル設定を読み込み: lgbm.objective={lgbm_params.get('objective')}")

    # データ準備と分割
    x, y = prepare_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, **split_params)

    # モデル構築・学習・予測
    model = LGBMRegressor(**lgbm_params)
    model.fit(x_train, y_train)
    y_pred_log = model.predict(x_test)

    # 評価と保存
    metrics = evaluate_regression(y_test, y_pred_log)
    log_metrics(metrics)
    save_model(model)

    logger.info(f"モデル学習完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
