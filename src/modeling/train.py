"""機械学習モデルを学習するモジュール."""

# 標準ライブラリ
import sys
import time
from pathlib import Path

import joblib

# サードパーティ
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
from src.utils.logger import get_logger  # noqa: E402

# モジュール定数
# 生データの配置先（プロジェクトルート基準の絶対パスで固定）
_DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

# モデル保存設定
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ロガーの初期化
logger = get_logger(__name__)


def evaluate_regression(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    """log価格予測モデルをlogスケール・円スケールの両方で評価する"""
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


def main() -> None:
    """エントリポイント."""
    start_time = time.time()

    # データの読み込み
    df = run_pipeline()

    df["取引価格（総額）"] = pd.to_numeric(df["取引価格（総額）"], errors="coerce")
    # 0以下の値は対数変換できないので削除
    df = df[df["取引価格（総額）"] > 0]
    df["log_取引価格"] = np.log(df["取引価格（総額）"])

    # データ分割
    leak_columns = ["取引価格（総額）", "log_取引価格"]
    X = df.drop(columns=leak_columns)
    y = df["log_取引価格"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # モデル構築
    model = LGBMRegressor(
        objective="regression", metric="rmse", n_estimators=1000,
        learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    # 学習・予測
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # 評価
    metrics = evaluate_regression(y_test, y_pred_log)

    logger.info(f"R2 log score: {metrics['r2_log']:.3f}")  # 価格のばらつきの説明具合(%)
    logger.info(f"RMSE log score: {metrics['rmse_log']:.3f}")  # 価格の二乗平均平方根誤差
    logger.info(f"MAE yen score: {metrics['mae_yen']:.0f}")  # 円スケールでの平均絶対誤差
    logger.info(f"RMSE yen score: {metrics['rmse_yen']:.0f}")  # 円スケールでの二乗平均平方根誤差
    logger.info(f"MAPE score: {metrics['mape']:.3f}")  # 平均絶対パーセント誤差
    logger.info(f"Median APE score: {metrics['median_ape']:.3f}")  # 中央絶対パーセント誤差

    # モデル保存
    joblib.dump(model, _MODEL_DIR / "lgbm_model.pkl")
    logger.info(f"モデルを保存: {_MODEL_DIR / 'lgbm_model.pkl'}")

    logger.info(f"モデル学習完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
