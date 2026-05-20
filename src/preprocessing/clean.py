"""データフレームの欠損値補完を行うモジュール."""

# 標準ライブラリ
import math
import re
import sys
import time
from pathlib import Path

# サードパーティ
import numpy as np
import pandas as pd

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# プロジェクト内モジュール（sys.path 操作後である必要があるため E402 を許容）
from src.utils.logger import get_logger  # noqa: E402
from src.utils.utils import load_data  # noqa: E402

# モジュール定数
# 前処理済みデータの保存先（プロジェクトルート基準の絶対パスで固定）
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# ロガーの初期化
logger = get_logger(__name__)

# 和暦→西暦変換用の基準年（元号1年 = base + 1 になる）
_WAREKI_BASE_YEAR = {
    "令和": 2018,  # 令和1年 = 2019
    "平成": 1988,  # 平成1年 = 1989
    "昭和": 1925,  # 昭和1年 = 1926
    "大正": 1911,  # 大正1年 = 1912
    "明治": 1867,  # 明治1年 = 1868
}
_WAREKI_PATTERN = re.compile(r"(令和|平成|昭和|大正|明治)(\d+)年")
_SEIREKI_PATTERN = re.compile(r"^(\d{4})")


def parse_construction_year(value: object) -> float:
    """建築年文字列を西暦の数値に変換する.

    国土交通省の不動産情報ライブラリでは建築年が和暦表記
    （例: ``平成5年``, ``昭和60年``）で記録されているため、
    数値計算可能な西暦に変換する。``"戦前"`` などの解釈不能な値は
    ``NaN`` として扱う。

    Args:
        value: 建築年の値。和暦・西暦・NaN のいずれかを想定。

    Returns:
        西暦の年数（float）。変換不能な場合は ``NaN``。
    """
    # NaN・None は NaN を返す
    if value is None:
        return float("nan")
    if isinstance(value, float) and math.isnan(value):
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    # 和暦表記の判定（例: "平成5年"）
    match = _WAREKI_PATTERN.match(text)
    if match:
        era, year_str = match.groups()
        return float(_WAREKI_BASE_YEAR[era] + int(year_str))

    # 既に西暦表記（例: "2010" や "2010年"）
    match_seireki = _SEIREKI_PATTERN.match(text)
    if match_seireki:
        return float(match_seireki.group(1))

    # "戦前" 等の解釈不能な値
    return float("nan")


def infer_residential_usage(df: pd.DataFrame, score_threshold: int = 2) -> pd.DataFrame:
    """用途が欠損している行を、他列から推定して『住宅』で補完する。

    判定スコア（各1点、合計が score_threshold 以上で住宅とみなす）:
      - 間取りが記入されている
      - 建物の構造が住宅系（ＲＣ／ＳＲＣ／木造／鉄骨造 など）
      - 種類が住宅系（中古マンション等 など）

    Args:
        df: 不動産情報ライブラリのDataFrame。
        score_threshold: 住宅とみなす最小スコア。デフォルト2。

    Returns:
        用途列を補完した新しいDataFrame。
    """
    # 副作用を避けるためコピー
    result = df.copy()

    # 用途列が無ければ補完対象がないのでそのまま返す
    if "用途" not in result.columns:
        logger.warning("'用途' 列が存在しないため infer_residential_usage をスキップします。")
        return result

    # 欠損行のマスク
    mask_missing = result["用途"].isna()

    # 列ごとに「存在すれば指標、無ければ全行 False」として扱うためのヘルパー
    def _flag_or_false(condition: pd.Series | None) -> pd.Series:
        if condition is None:
            return pd.Series(False, index=result.index)
        return condition.fillna(False)

    # 各住宅指標（True=住宅らしい）。参照する列が無ければその指標は使わない
    has_madori = _flag_or_false(
        result["間取り"].notna() if "間取り" in result.columns else None
    )

    residential_structures = ["ＲＣ", "ＳＲＣ", "木造", "鉄骨造", "軽量鉄骨造", "ブロック造"]
    has_residential_structure = _flag_or_false(
        result["建物の構造"].isin(residential_structures)
        if "建物の構造" in result.columns
        else None
    )

    residential_types = ["中古マンション等", "宅地(建物)"]
    has_residential_type = _flag_or_false(
        result["種類"].isin(residential_types) if "種類" in result.columns else None
    )

    # 参照できなかった列があればログに警告
    missing_cols = [
        col for col in ("間取り", "建物の構造", "種類") if col not in result.columns
    ]
    if missing_cols:
        logger.warning(
            f"infer_residential_usage: 以下の列が無いため指標から除外: {missing_cols}"
        )

    # スコア合算
    score = (
        has_madori.astype(int)
        + has_residential_structure.astype(int)
        + has_residential_type.astype(int)
    )
    is_likely_residential = score >= score_threshold

    # 「欠損 かつ 住宅らしい」行だけを補完
    fill_mask = mask_missing & is_likely_residential
    result.loc[fill_mask, "用途"] = "住宅"

    # 補完結果のログ
    n_filled = int(fill_mask.sum())
    n_remaining = int(mask_missing.sum() - n_filled)
    logger.info(f"用途を住宅と推定して補完: {n_filled:,} 件")
    logger.info(f"用途の判定不能のため欠損のまま: {n_remaining:,} 件")

    return result


def fill_zoning_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """建蔽率・容積率を立地ベースで階層的に補完する。

    補完の優先順位:
        1. 同一地区名の最頻値（最も近い物件群で補完）
        2. 同一市区町村名の最頻値（地区不明の場合）
        3. 全体の最頻値（最終フォールバック）

    Args:
        df: 不動産情報ライブラリのDataFrame。

    Returns:
        建蔽率・容積率を補完した新しいDataFrame。
    """
    result = df.copy()
    target_cols = ["建ぺい率（％）", "容積率（％）"]

    for col in target_cols:
        # 1. 地区名グループの最頻値で埋める
        result[col] = result.groupby("地区名")[col].transform(
            lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s
        )
        # 2. まだ欠損なら市区町村名グループの最頻値で埋める
        result[col] = result.groupby("市区町村名")[col].transform(
            lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s
        )
        # 3. それでも欠損なら全体の最頻値で埋める
        global_mode = result[col].mode().iloc[0]
        result[col] = result[col].fillna(global_mode)

        n_remaining = int(result[col].isna().sum())
        logger.info(f"{col}: 補完後の欠損数 = {n_remaining:,}")

    return result


def refine_data(df: pd.DataFrame) -> pd.DataFrame:
    """データのクリーニングを行う.

    - 不要な列の削除
    - 欠損値の処理
    - データ型の変換

    Args:
        df: ロードしたデータフレーム。

    Returns:
        クリーニング後のデータフレーム。
    """
    logger.info("Starting data cleaning...")

    # 例: 不要な列 'ID' を削除
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        logger.info("Dropped column 'ID'.")

    # if "種類" in df.columns:
    #     df = df.drop(columns=["種類"])
    #     logger.info("Dropped column '種類'.")

    if "最寄駅：名称" in df.columns:
        missing_station_count = df["最寄駅：名称"].isna().sum()
        logger.info(f"Found {missing_station_count} missing values in '最寄駅：名称' column.")
        # 欠損値を 'Unknown' で埋める
        df["最寄駅：名称"] = df["最寄駅：名称"].fillna("Unknown")
        logger.info("Filled missing values in '最寄駅：名称' with 'Unknown'.")

    if "最寄駅：距離（分）" in df.columns:
        df["最寄駅：距離（分）"] = pd.to_numeric(df["最寄駅：距離（分）"], errors="coerce")
        missing_distance_count = df["最寄駅：距離（分）"].isna().sum()
        logger.info(
            f"Found {missing_distance_count} missing values in '最寄駅：距離（分）' column."
        )
        # 欠損値を最頻値で埋める
        mode_distance = df["最寄駅：距離（分）"].mode()[0]
        df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].fillna(mode_distance)
        logger.info(
            f"Filled missing values in '最寄駅：距離（分）' with mode value {mode_distance}."
        )

    if "間取り" in df.columns:
        missing_layout_count = df["間取り"].isna().sum()
        logger.info(f"Found {missing_layout_count} missing values in '間取り' column.")
        # 欠損値を 'Unknown' で埋める
        df["間取り"] = df["間取り"].fillna("Unknown")
        logger.info("Filled missing values in '間取り' with 'Unknown'.")

    if "建築年" in df.columns:
        # 和暦表記（"平成5年" 等）を西暦の数値に変換。"戦前" 等は NaN になる
        df["建築年"] = df["建築年"].map(parse_construction_year)
        missing_construction_year_count = int(df["建築年"].isna().sum())
        logger.info(f"Found {missing_construction_year_count} missing values in '建築年' column.")
        # 欠損値を中央値で埋める
        median_construction_year = df["建築年"].median()
        df["建築年"] = df["建築年"].fillna(median_construction_year)
        logger.info(f"Filled missing values in '建築年' with median value {median_construction_year}.")

        # 築年数を計算（西暦に変換済みなので数値演算可能）
        current_year = pd.Timestamp.now().year
        df["築年数"] = current_year - df["建築年"]
        logger.info("Calculated '築年数' from '建築年'.")

    # 注: 「築年数」のNaN補完は不要。建築年を先にmedian補完してから引き算するため。

    if "建物の構造" in df.columns:
        missing_structure_count = df["建物の構造"].isna().sum()
        logger.info(f"Found {missing_structure_count} missing values in '建物の構造' column.")
        # 欠損値を 'Unknown' で埋める
        df["建物の構造"] = df["建物の構造"].fillna("Unknown")
        logger.info("Filled missing values in '建物の構造' with 'Unknown'.")

    if "用途" in df.columns:
        missing_usage_count = df["用途"].isna().sum()
        logger.info(f"Found {missing_usage_count} missing values in '用途' column.")
        # 用途が欠損している行を推定して補完
        df = infer_residential_usage(df)

    if "今後の利用目的" in df.columns:
        missing_feature_count = df["今後の利用目的"].isna().sum()
        logger.info(f"Found {missing_feature_count} missing values in '今後の利用目的' column.")
        # 欠損値を最頻値で補完(クロス集計結果から)
        most_common_purpose = df["今後の利用目的"].mode()[0]
        df["今後の利用目的"] = df["今後の利用目的"].fillna(most_common_purpose)
        logger.info(
            f"Filled missing values in '今後の利用目的' with most common value '{most_common_purpose}'."
        )

    if "都市計画" in df.columns:
        missing_urban_planning_count = df["都市計画"].isna().sum()
        logger.info(f"Found {missing_urban_planning_count} missing values in '都市計画' column.")
        # 欠損値を 'Unknown' で埋める
        df["都市計画"] = df["都市計画"].fillna("Unknown")
        logger.info("Filled missing values in '都市計画' with 'Unknown'.")

    # 建蔽率・容積率を立地ベースで補完（必要な列がすべて揃っている時のみ実行）
    zoning_required_cols = ["建ぺい率（％）", "容積率（％）", "地区名", "市区町村名"]
    if all(col in df.columns for col in zoning_required_cols):
        df = fill_zoning_ratios(df)

    if "改装" in df.columns:
        missing_renovation_count = df["改装"].isna().sum()
        logger.info(f"Found {missing_renovation_count} missing values in '改装' column.")
        # 欠損値を 'Unknown' で埋める
        df["改装"] = df["改装"].fillna("Unknown")
        logger.info("Filled missing values in '改装' with 'Unknown'.")

    if "取引の事情等" in df.columns:
        # 列を削除
        df = df.drop(columns=["取引の事情等"])
        logger.info("Dropped column '取引の事情等'.")

    # 例: 日付列 'Date' を datetime 型に変換
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        logger.info("Converted 'Date' column to datetime.")

    logger.info("Data cleaning completed.")
    return df


def outlier_handling(df: pd.DataFrame) -> pd.DataFrame:
    """外れ値の処理を行う.

    例: 取引価格（総額）が極端に高い・低い行を削除するなど。

    Args:
        df: クリーニングされたデータフレーム。

    Returns:
        外れ値処理後のデータフレーム。
    """

    # 取引総額（価格） を対数変換
    if "取引価格（総額）" in df.columns:
        df["取引価格（総額）"] = pd.to_numeric(df["取引価格（総額）"], errors="coerce")
        # 0以下の値は対数変換できないので削除
        df = df[df["取引価格（総額）"] > 0]
        df["取引価格（総額）"] = np.log(df["取引価格（総額）"])
        logger.info("Applied log transformation to '取引価格（総額）' and removed non-positive values.")

    # 300㎡を超える 面積（㎡） は異常値とみなして削除
    if "面積（㎡）" in df.columns:
        initial_count = len(df)
        df = df[df["面積（㎡）"] <= 300]
        final_count = len(df)
        logger.info(f"Removed outliers in '面積（㎡）': {initial_count - final_count} rows removed.")

    # 建ぺい率（％）
    if "建ぺい率（％）" in df.columns:
        initial_count = len(df)
        df = df[df["建ぺい率（％）"] <= 600.0]
        final_count = len(df)
        logger.info(f"Removed outliers in '建ぺい率（％）': {initial_count - final_count} rows removed.")

    # 容積率（％）: 60.0 <= x <= 800.0で採用
    if "容積率（％）" in df.columns:
        initial_count = len(df)
        df = df[(df["容積率（％）"] >= 60.0) & (df["容積率（％）"] <= 800.0)]
        final_count = len(df)
        logger.info(f"Removed outliers in '容積率（％）': {initial_count - final_count} rows removed.")

    # 築年数の負値（建築年が未来の前売り物件）は 0 にクリップする
    if "築年数" in df.columns:
        n_negative = int((df["築年数"] < 0).sum())
        if n_negative > 0:
            df["築年数"] = df["築年数"].clip(lower=0)
            logger.info(f"Clipped {n_negative} negative '築年数' values to 0.")

    # 完全重複行を削除する
    initial_count = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_count - len(df)} duplicate rows.")

    return df


def preprocess_data(file_path: str) -> pd.DataFrame:
    """データの前処理を一括で実行する関数.

    Args:
        file_path: CSVファイルのパス。

    Returns:
        前処理されたデータフレーム。
    """
    start_time = time.time()
    fill_nan_df = load_data(file_path)
    df = refine_data(fill_nan_df)
    df = outlier_handling(df)
    # 出力先ディレクトリを自動作成して、プロジェクトルート基準で保存する
    # （notebook / CLI など実行場所に依存せず常に同じ場所に出力される）
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _PROCESSED_DIR / "cleaned_data.csv"
    df.to_csv(output_path, index=False, encoding="cp932")
    logger.info(f"Saved cleaned data to {output_path}")

    end_time = time.time()
    logger.info(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")

    return df
