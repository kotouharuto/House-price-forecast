"""機械学習モデルを学習するモジュール."""

# 標準ライブラリ
import sys
import time
from pathlib import Path

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

# ロガーの初期化
logger = get_logger(__name__)


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
    model = LGBMRegressor()

    # 学習・予測
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # logスケール(モデルの素の性能)
    r2 = r2_score(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))  # ≒ RMSLE

    # 元スケール(円) - 解釈用
    y_test_yen = np.exp(y_test)
    y_pred_yen = np.exp(y_pred_log)
    rmse_yen = np.sqrt(mean_squared_error(y_test_yen, y_pred_yen))
    mae_yen = mean_absolute_error(y_test_yen, y_pred_yen)
    mape = np.mean(np.abs((y_test_yen - y_pred_yen) / y_test_yen)) * 100

    logger.info(f"R2 score: {r2:.3f}")
    logger.info(f"RMSE log score: {rmse_log:.3f}")
    logger.info(f"rmse_yen score: {rmse_yen:.3f}")
    logger.info(f"mae_yen score: {mae_yen:.3f}")
    logger.info(f"mape score: {mape:.3f}")

    logger.info(f"モデル学習完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
