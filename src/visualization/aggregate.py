"""予測結果を行政区単位・駅単位で集計するモジュール.

BIツール（``app/pages/``）から利用する、UIに依存しない純粋な集計ロジック。
予測結果CSV（``outputs/test_predictions.csv``）を読み込み、地図・グラフ表示用の
集計テーブルを生成する。Streamlit等のUI依存を持たないため、単体でテスト可能。
"""

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# プロジェクトルート基準の絶対パス（呼び出し場所に依存しないようにする）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PRED_PATH = _PROJECT_ROOT / "outputs" / "test_predictions.csv"

# 入力CSVパスを切り替える環境変数（別モデル・別出力の確認用）
_PRED_PATH_ENV_VAR = "BI_PREDICTIONS_PATH"

# 予測結果CSVの列名（マジック文字列を避けるため定数化）
ACTUAL_PRICE_COL = "actual_price_yen"
PRED_PRICE_COL = "pred_price_yen"
ERROR_YEN_COL = "error_yen"
ERROR_RATE_COL = "error_rate_percent"
APE_COL = "ape_percent"
PRICE_BAND_COL = "actual_price_band"
WARD_CODE_COL = "市区町村コード"
ADDRESS_COL = "住所"
STATION_COL = "最寄駅：名称"
STATION_LAT_COL = "最寄駅：緯度"
STATION_LON_COL = "最寄駅：経度"
TYPE_COL = "種類"
AREA_COL = "面積（㎡）"
AGE_COL = "築年数"
CITY_PLAN_COL = "都市計画"
STATION_DISTANCE_COL = "最寄駅：距離（分）"
YAMANOTE_COL = "山手線内側"
ACTUAL_LOG_COL = "actual_log_price"
PRED_LOG_COL = "pred_log_price"

# 読み込み時に存在を検証する必須列（ダッシュボードが実際に参照する列に限定）。
# log列（actual_log_price/pred_log_price）は summarize_metrics が欠損を許容する
# 設計のため、また 住所・都市計画 等の未使用列は正常CSVを弾かないため含めない。
_REQUIRED_COLUMNS: tuple[str, ...] = (
    ACTUAL_PRICE_COL,
    PRED_PRICE_COL,
    ERROR_YEN_COL,
    APE_COL,
    PRICE_BAND_COL,
    WARD_CODE_COL,
    STATION_COL,
    STATION_LAT_COL,
    STATION_LON_COL,
    TYPE_COL,
    AREA_COL,
    AGE_COL,
    YAMANOTE_COL,
)


def default_predictions_path() -> Path:
    """予測結果CSVの既定パスを返す.

    環境変数 ``BI_PREDICTIONS_PATH`` が設定されていればそのパスを、
    未設定なら ``outputs/test_predictions.csv`` を返す。別モデル・別出力の
    結果を確認する際に、コード変更なしで入力を切り替えられるようにする。

    Returns:
        予測結果CSVのパス。
    """
    env_path = os.environ.get(_PRED_PATH_ENV_VAR)
    return Path(env_path) if env_path else _DEFAULT_PRED_PATH


def load_predictions(file_path: str | Path | None = None) -> pd.DataFrame:
    """予測結果CSVを読み込み、必須列の存在を検証する.

    Args:
        file_path: 予測結果CSVのパス。``None`` の場合は
            :func:`default_predictions_path`（環境変数または既定パス）を使う。

    Returns:
        予測結果のデータフレーム。

    Raises:
        FileNotFoundError: 指定したCSVファイルが存在しない場合。
        KeyError: 集計に必要な列が欠落している場合。
    """
    path = Path(file_path) if file_path is not None else default_predictions_path()
    if not path.exists():
        logger.error(f"予測結果ファイルが見つかりません: {path}")
        raise FileNotFoundError(f"予測結果ファイルが見つかりません: {path}")

    df = pd.read_csv(path)

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"予測結果CSVに必須列がありません: {missing}")
        raise KeyError(f"予測結果CSVに必須列がありません: {missing}")

    logger.info(f"予測結果を読み込みました: shape={df.shape}, path={path}")
    return df


def aggregate_predictions(
    df: pd.DataFrame,
    group_col: str,
    coord_cols: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """指定キーで予測結果を集計し、地図・グラフ用の集計テーブルを返す.

    各グループについて件数・予測価格/実測価格の平均と中央値・APEの平均(MAPE)と
    中央値(Median APE)を算出する。``coord_cols`` を指定すると、地図配置用の
    代表座標（グループ平均）を ``lat`` / ``lon`` 列として付与する。

    Args:
        df: 予測結果のデータフレーム。
        group_col: 集計キーとなる列名（例: ``"市区町村コード"`` / ``"最寄駅：名称"``）。
        coord_cols: 代表座標を計算する ``(緯度列, 経度列)``。不要なら ``None``。

    Returns:
        集計結果のデータフレーム。``group_col`` と集計列を持つ。
        列: ``count``, ``pred_price_mean``, ``pred_price_median``,
        ``actual_price_mean``, ``actual_price_median``, ``mape``, ``median_ape``
        （``coord_cols`` 指定時は ``lat``, ``lon`` を追加）。

    Raises:
        KeyError: ``group_col`` または ``coord_cols`` の列が存在しない場合。
    """
    if group_col not in df.columns:
        logger.error(f"集計キー列が存在しません: {group_col}")
        raise KeyError(f"集計キー列が存在しません: {group_col}")

    # observed=True: category型キーでも実在する値のみを集計対象にする
    grouped = df.groupby(group_col, observed=True)

    result = grouped.agg(
        count=(PRED_PRICE_COL, "size"),
        pred_price_mean=(PRED_PRICE_COL, "mean"),
        pred_price_median=(PRED_PRICE_COL, "median"),
        actual_price_mean=(ACTUAL_PRICE_COL, "mean"),
        actual_price_median=(ACTUAL_PRICE_COL, "median"),
        mape=(APE_COL, "mean"),
        median_ape=(APE_COL, "median"),
    )

    # 地図配置用に代表緯度経度（グループ平均、NaNは自動除外）を付与
    if coord_cols is not None:
        lat_col, lon_col = coord_cols
        missing_coords = [col for col in coord_cols if col not in df.columns]
        if missing_coords:
            logger.error(f"座標列が存在しません: {missing_coords}")
            raise KeyError(f"座標列が存在しません: {missing_coords}")

        coords = grouped[[lat_col, lon_col]].mean().rename(columns={lat_col: "lat", lon_col: "lon"})
        result = result.join(coords)

    logger.info(f"集計完了: key={group_col}, groups={len(result):,}")
    return result.reset_index()


def aggregate_by_ward(df: pd.DataFrame) -> pd.DataFrame:
    """行政区（市区町村コード）単位で予測結果を集計する.

    Args:
        df: 予測結果のデータフレーム。

    Returns:
        市区町村コードをキーとする集計データフレーム。
    """
    return aggregate_predictions(df, WARD_CODE_COL)


def aggregate_by_station(df: pd.DataFrame) -> pd.DataFrame:
    """最寄駅単位で予測結果を集計し、代表座標を付与する.

    Args:
        df: 予測結果のデータフレーム。

    Returns:
        最寄駅名をキーとし、地図配置用の ``lat`` / ``lon`` を含む集計データフレーム。
    """
    return aggregate_predictions(df, STATION_COL, coord_cols=(STATION_LAT_COL, STATION_LON_COL))


def _representative_category(series: pd.Series) -> object:
    """系列の最頻値を返す（同数なら先頭、空なら ``NaN``）."""
    mode = series.mode(dropna=True)
    return mode.iat[0] if not mode.empty else np.nan


def station_map_summary(df: pd.DataFrame) -> pd.DataFrame:
    """地図ポップアップ用に、駅単位の集計へ代表的な物件種類・価格帯を付与する.

    ``aggregate_by_station`` の集計（件数・価格・APE・代表座標）に、駅ごとの
    最頻の物件種類と価格帯を ``repr_type`` / ``repr_band`` として結合する。

    Args:
        df: 予測結果のデータフレーム。

    Returns:
        駅単位の集計に ``repr_type`` / ``repr_band`` を加えたデータフレーム。
    """
    agg = aggregate_by_station(df)
    representative = (
        df.groupby(STATION_COL, observed=True)
        .agg(
            repr_type=(TYPE_COL, _representative_category),
            repr_band=(PRICE_BAND_COL, _representative_category),
        )
        .reset_index()
    )
    return agg.merge(representative, on=STATION_COL, how="left")


def filter_predictions(
    df: pd.DataFrame,
    ward_codes: Sequence[int] | None = None,
    stations: Sequence[str] | None = None,
    property_types: Sequence[str] | None = None,
    price_bands: Sequence[str] | None = None,
    area_range: tuple[float, float] | None = None,
    age_range: tuple[float, float] | None = None,
    yamanote_inside: bool | None = None,
) -> pd.DataFrame:
    """サイドバーのフィルタ条件で予測結果を絞り込む.

    各引数は ``None`` または空の場合は当該条件を適用しない。指定された条件のみを
    AND で結合してマスクを構築する純粋関数。

    Args:
        df: 予測結果のデータフレーム。
        ward_codes: 対象の市区町村コード（複数選択）。
        stations: 対象の最寄駅名（複数選択）。
        property_types: 対象の物件種類（複数選択）。
        price_bands: 対象の価格帯ラベル（複数選択）。
        area_range: 面積（㎡）の ``(下限, 上限)``。
        age_range: 築年数の ``(下限, 上限)``。
        yamanote_inside: ``True`` で山手線内側のみ、``False`` で外側のみ。

    Returns:
        条件で絞り込んだデータフレームのコピー。
    """
    mask = pd.Series(True, index=df.index)

    if ward_codes:
        mask &= df[WARD_CODE_COL].isin(list(ward_codes))
    if stations:
        mask &= df[STATION_COL].isin(list(stations))
    if property_types:
        mask &= df[TYPE_COL].isin(list(property_types))
    if price_bands:
        mask &= df[PRICE_BAND_COL].isin(list(price_bands))
    if area_range is not None:
        mask &= df[AREA_COL].between(*area_range)
    if age_range is not None:
        mask &= df[AGE_COL].between(*age_range)
    if yamanote_inside is not None:
        mask &= df[YAMANOTE_COL] == int(yamanote_inside)

    return df[mask].copy()


def available_stations(
    df: pd.DataFrame,
    ward_codes: Sequence[int] | None = None,
) -> list[str]:
    """指定した行政区内に存在する最寄駅名の一覧を昇順で返す.

    行政区フィルタと最寄駅フィルタを連動させるためのヘルパー。``ward_codes`` が
    空または ``None`` の場合は全駅を返す。

    Args:
        df: 予測結果のデータフレーム。
        ward_codes: 対象の市区町村コード。指定時はその区内の駅のみに絞る。

    Returns:
        最寄駅名の昇順リスト（重複・欠損を除外）。
    """
    pool = df if not ward_codes else df[df[WARD_CODE_COL].isin(list(ward_codes))]
    return sorted(pool[STATION_COL].dropna().unique().tolist())


def price_band_order(df: pd.DataFrame) -> list[str]:
    """価格帯ラベルを金額順（各帯の平均実測価格の昇順）に並べて返す.

    ``"~2000万"`` / ``"3億~"`` のようにラベルの表記が一定でないため、文字列の
    パースではなく、各価格帯に属する実測価格の平均値でソートして安定した
    金額順を得る。フィルタの選択肢順を整えるために用いる。

    Args:
        df: 予測結果のデータフレーム。

    Returns:
        金額昇順に並べた価格帯ラベルのリスト。
    """
    order = (
        df.groupby(PRICE_BAND_COL, observed=True)[ACTUAL_PRICE_COL]
        .mean()
        .sort_values()
        .index.tolist()
    )
    return [str(band) for band in order]


def summarize_metrics(df: pd.DataFrame) -> dict[str, float]:
    """予測結果全体の評価指標サマリを算出する（KPI表示用）.

    円スケールの MAE・RMSE、APE の平均(MAPE)と中央値(Median APE) を計算する。
    logスケールの列が存在する場合は R²(log) も併せて算出する。空のデータフレーム
    では件数 0・各指標 NaN を返し、ゼロ除算を避ける。

    Args:
        df: 予測結果のデータフレーム。

    Returns:
        ``count``, ``r2_log``, ``mae_yen``, ``rmse_yen``, ``mape``,
        ``median_ape`` をキーとする辞書。
    """
    n = len(df)
    if n == 0:
        nan = float("nan")
        return {
            "count": 0,
            "r2_log": nan,
            "mae_yen": nan,
            "rmse_yen": nan,
            "mape": nan,
            "median_ape": nan,
        }

    actual = df[ACTUAL_PRICE_COL].to_numpy(dtype=float)
    pred = df[PRED_PRICE_COL].to_numpy(dtype=float)
    ape = df[APE_COL].to_numpy(dtype=float)
    error = actual - pred

    # logスケールの列が揃っていれば決定係数 R² を計算する
    r2_log = float("nan")
    if {ACTUAL_LOG_COL, PRED_LOG_COL}.issubset(df.columns):
        y_true = df[ACTUAL_LOG_COL].to_numpy(dtype=float)
        y_pred = df[PRED_LOG_COL].to_numpy(dtype=float)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "count": n,
        "r2_log": r2_log,
        "mae_yen": float(np.mean(np.abs(error))),
        "rmse_yen": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
    }
