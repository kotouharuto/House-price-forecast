"""機械学習モデルを学習するモジュール.

Optuna でハイパーパラメータをチューニングし、最良パラメータで最終モデルを学習する。
固定パラメータ・探索範囲・Optuna設定は ``configs/model_params.yaml`` で管理し、
``src.utils.config.load_model_params()`` 経由で読み込む。
"""

# 標準ライブラリ
import sys
import time
from pathlib import Path
from typing import Any

# サードパーティ
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from optuna.integration import LightGBMPruningCallback
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

# モジュール定数
_TARGET_COLUMN = "取引価格（総額）"
_LOG_TARGET_COLUMN = "log_取引価格"

# 学習側の外れ値処理: 高額物件の影響で MAE/RMSE が膨らむため、学習・検証データの
# 目的変数のみ上限値で頭打ち（winsorize）にする。test は素のまま評価する。
_TRAIN_PRICE_CAP_YEN = 3e8
_TRAIN_LOG_PRICE_CAP = float(np.log(_TRAIN_PRICE_CAP_YEN))

# モデル保存設定
_MODEL_DIR = _PROJECT_ROOT / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 結果分析用CSVの保存先
_OUTPUT_DIR = _PROJECT_ROOT / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_PREDICTION_ANALYSIS_PATH = _OUTPUT_DIR / "test_predictions.csv"

# ロガーの初期化
logger = get_logger(__name__)


def suggest_params(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
    fixed_params: dict[str, Any],
) -> dict[str, Any]:
    """Optunaのtrialから、探索範囲に基づいてパラメータを生成する.

    Args:
        trial: Optunaのトライアル。
        search_space: 探索範囲の定義（model_params.yaml の lgbm_search_space）。
        fixed_params: 固定パラメータ（model_params.yaml の lgbm）。

    Returns:
        固定パラメータと探索パラメータをマージした辞書。
    """
    params = dict(fixed_params)
    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    return params


def objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    search_space: dict[str, dict[str, Any]],
    fixed_params: dict[str, Any],
) -> float:
    """Optunaの目的関数。検証セットのRMSE（logスケール）を最小化する.

    Args:
        trial: Optunaのトライアル。
        x_train, y_train: 学習データ。
        x_val, y_val: 検証データ。
        search_space: 探索範囲の定義。
        fixed_params: 固定パラメータ。

    Returns:
        検証セットのRMSE（logスケール）。
    """
    params = suggest_params(trial, search_space, fixed_params)
    model = LGBMRegressor(**params)
    # 見込みのない試行を学習途中で打ち切るための枝刈りコールバック
    pruning_callback = LightGBMPruningCallback(trial, "rmse")
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), pruning_callback],
    )
    y_pred = model.predict(x_val)
    return float(np.sqrt(mean_squared_error(y_val, y_pred)))


def evaluate_regression(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    """log価格予測モデルをlogスケール・円スケールの両方で評価する.

    Args:
        y_true_log: 正解値（logスケール）。
        y_pred_log: 予測値（logスケール）。

    Returns:
        各種評価指標の辞書。
    """
    y_true_yen = np.exp(y_true_log)
    y_pred_yen = np.exp(y_pred_log)

    abs_error_yen = np.abs(y_true_yen - y_pred_yen)
    ape = abs_error_yen / y_true_yen * 100

    return {
        "r2_log": r2_score(y_true_log, y_pred_log),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_yen": mean_absolute_error(y_true_yen, y_pred_yen),
        "rmse_yen": float(np.sqrt(mean_squared_error(y_true_yen, y_pred_yen))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
    }


def create_prediction_analysis_df(
    source_df: pd.DataFrame,
    x_test: pd.DataFrame,
    y_true_log: pd.Series,
    y_pred_log: np.ndarray,
) -> pd.DataFrame:
    """元データにテストデータの予測値・誤差を付与して分析用にまとめる.

    Args:
        source_df: 目的変数を含む元データ。
        x_test: テストデータの特徴量。
        y_true_log: 正解値（logスケール）。
        y_pred_log: 予測値（logスケール）。

    Returns:
        分析用のDataFrame。価格は円スケールに復元済み。
    """
    y_true_yen = np.exp(y_true_log.to_numpy())
    y_pred_yen = np.exp(y_pred_log)
    error_yen = y_pred_yen - y_true_yen
    abs_error_yen = np.abs(error_yen)
    error_rate = error_yen / y_true_yen * 100
    ape = abs_error_yen / y_true_yen * 100

    base_df = source_df.loc[x_test.index].copy().reset_index(names="original_index")
    prediction_df = pd.DataFrame(
        {
            "actual_log_price": y_true_log.to_numpy(),
            "pred_log_price": y_pred_log,
            "actual_price_yen": np.rint(y_true_yen).astype("int64"),
            "pred_price_yen": np.rint(y_pred_yen).astype("int64"),
            "error_yen": np.rint(error_yen).astype("int64"),
            "abs_error_yen": np.rint(abs_error_yen).astype("int64"),
            "error_rate_percent": error_rate,
            "ape_percent": ape,
        }
    )
    prediction_df["actual_price_band"] = pd.cut(
        prediction_df["actual_price_yen"],
        bins=[0, 20_000_000, 50_000_000, 100_000_000, 300_000_000, np.inf],
        labels=["~2000万", "2000万~5000万", "5000万~1億", "1億~3億", "3億~"],
    )

    return pd.concat([base_df, prediction_df], axis=1)


def save_prediction_analysis(
    source_df: pd.DataFrame,
    x_test: pd.DataFrame,
    y_true_log: pd.Series,
    y_pred_log: np.ndarray,
    output_path: Path = _PREDICTION_ANALYSIS_PATH,
) -> None:
    """テストデータの予測結果を分析用CSVとして保存する."""
    analysis_df = create_prediction_analysis_df(source_df, x_test, y_true_log, y_pred_log)
    analysis_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"予測結果の分析用CSVを保存: {output_path}")


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """前処理・特徴量エンジニアリングを実行し、元データ・特徴量X・目的変数yを返す.

    目的変数は log 変換する。取引価格列はリーク防止のため X から除外する。
    学習側だけの外れ値クリップは ``main()`` の train_test_split 後に行うため、
    本関数では行わない（test 側を素のまま評価できるようにするため）。

    Returns:
        (df, X, y) のタプル。df は目的変数を含む元データ、y は log 変換済みの取引価格。
    """
    df = run_pipeline()

    df[_TARGET_COLUMN] = pd.to_numeric(df[_TARGET_COLUMN], errors="coerce")
    # 0以下は対数変換できないため除外
    df = df[df[_TARGET_COLUMN] > 0]
    df[_LOG_TARGET_COLUMN] = np.log(df[_TARGET_COLUMN])

    # 目的変数（生・log両方）はリークになるため特徴量から除外
    x = df.drop(columns=[_TARGET_COLUMN, _LOG_TARGET_COLUMN])
    y = df[_LOG_TARGET_COLUMN]
    return df, x, y


def save_model(model: LGBMRegressor, best_params: dict[str, Any]) -> None:
    """学習済みモデルと最良パラメータを models/ に保存する.

    Args:
        model: 学習済みモデル。
        best_params: 最良ハイパーパラメータ。
    """
    model_path = _MODEL_DIR / "lgbm_model.pkl"
    params_path = _MODEL_DIR / "best_params.yaml"

    joblib.dump(model, model_path)
    with params_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, allow_unicode=True, sort_keys=False)

    logger.info(f"モデルを保存: {model_path}")
    logger.info(f"最良パラメータを保存: {params_path}")


def main() -> None:
    """エントリポイント: チューニング → 最終学習 → 評価 → 保存."""
    start_time = time.time()

    # 設定の読み込み（固定パラメータ・探索範囲・Optuna設定）
    fixed_params = load_model_params("lgbm")
    search_space = load_model_params("lgbm_search_space")
    optuna_config = load_model_params("optuna")

    # データ準備
    source_df, x, y = prepare_dataset()

    # train / val / test に分割
    test_size = optuna_config["test_size"]
    val_size = optuna_config["val_size"]
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    # 残りから val を切り出す（全体に対する val_size になるよう比率を調整）
    val_ratio = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=val_ratio, random_state=42
    )
    logger.info(f"データ分割: train={len(x_train):,}, val={len(x_val):,}, test={len(x_test):,}")

    # 学習・検証データの目的変数のみ上限値で頭打ちにする（test は素のまま評価）
    # 高額物件の極端な外れ値で MAE/RMSE が引きずられる問題への対処
    n_clipped_train = int((y_train > _TRAIN_LOG_PRICE_CAP).sum())
    n_clipped_val = int((y_val > _TRAIN_LOG_PRICE_CAP).sum())
    y_train = y_train.clip(upper=_TRAIN_LOG_PRICE_CAP)
    y_val = y_val.clip(upper=_TRAIN_LOG_PRICE_CAP)
    logger.info(
        f"学習側クリップ（上限={_TRAIN_PRICE_CAP_YEN / 1e8:.1f}億円）: "
        f"train={n_clipped_train}件, val={n_clipped_val}件を頭打ち（test は無修正）"
    )

    # Optuna でハイパーパラメータをチューニング
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # MedianPruner: 各試行の中間スコアが過去試行の中央値より悪ければ早期に打ち切る
    # n_warmup_steps: 最初の20イテレーションは打ち切らない（序盤の不安定さを許容）
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_val, y_val, search_space, fixed_params),
        n_trials=optuna_config["n_trials"],
    )
    logger.info(f"Optuna 完了: best RMSE_log={study.best_value:.4f}")
    logger.info(f"best_params={study.best_params}")

    # 最良パラメータで最終モデルを学習
    best_params = {**fixed_params, **study.best_params}
    final_model = LGBMRegressor(**best_params)
    final_model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # テストセットで評価
    y_pred_log = final_model.predict(x_test)
    metrics = evaluate_regression(y_test, y_pred_log)

    logger.info(f"R2 log score    : {metrics['r2_log']:.3f}")  # 価格のばらつきの説明具合
    logger.info(f"RMSE log score  : {metrics['rmse_log']:.3f}")  # logスケールのRMSE
    logger.info(f"MAE yen score   : {metrics['mae_yen']:,.0f} 円")  # 円スケールの平均絶対誤差
    logger.info(f"RMSE yen score  : {metrics['rmse_yen']:,.0f} 円")  # 円スケールのRMSE
    logger.info(f"MAPE score      : {metrics['mape']:.3f} %")  # 平均絶対パーセント誤差
    logger.info(f"Median APE score: {metrics['median_ape']:.3f} %")  # 中央絶対パーセント誤差

    # 結果分析用に、元データへlog予測値・円スケール予測値を付与してCSV出力
    save_prediction_analysis(source_df, x_test, y_test, y_pred_log)

    # モデルと最良パラメータを保存
    save_model(final_model, best_params)

    logger.info(f"モデル学習完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
