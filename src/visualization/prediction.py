"""予測ユーティリティ（点推定 + 予測区間）モジュール.

説明可能性ロードマップ Phase 2 の中核 API。Quantile Regression で学習した
3 つのモデル（low / median / high）から、対象物件の予測区間を返す。

不動産 AVM における「点推定 + 区間」のセット提示は、業務リスク管理
（融資判断・自動承認の閾値）や顧客説明（誠実な不確実性表示）の基盤となる。
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.utils.logger import get_logger
from src.visualization.similar_properties import find_similar_properties

logger = get_logger(__name__)

# 予測区間の DataFrame 列名（顧客向け UI からも参照される想定で定数化）
LOWER_COL = "lower"
MEDIAN_COL = "median"
UPPER_COL = "upper"

# 乖離度テーブルの列名（CSV 出力・評価で参照されるため定数化）
DIVERGENCE_COL = "divergence"
RATIO_COL = "ratio"

# 中央値乖離度の既定の「高い」判定閾値（compare_intervals と揃える）
DEFAULT_DIVERGENCE_THRESHOLD = 0.30

# 重大度バンドの境界（fold = 1.0 からの倍率乖離。(境界未満, ラベル) の昇順）
_BAND_BOUNDARIES: tuple[tuple[float, str], ...] = ((1.30, "軽度"), (2.00, "中度"))
_SEVERE_BAND = "重度"

# 想定する分位点数（low / median / high の 3 つ）
_EXPECTED_NUM_QUANTILES = 3


def predict_with_interval(
    models: dict[float, LGBMRegressor],
    x: pd.DataFrame,
    return_yen: bool = True,
) -> pd.DataFrame:
    """分位点回帰モデル群を使って予測区間付きの予測値を返す.

    α を昇順に並べ、最小・中央・最大の 3 つを ``lower`` / ``median`` /
    ``upper`` 列にマッピングする。

    Args:
        models: ``{alpha: 学習済み LGBMRegressor}`` の辞書。3 件必要。
        x: 予測対象の特徴量 DataFrame。
        return_yen: True なら log 予測を ``np.exp`` で円換算する。
            False なら log スケールのまま返す。

    Returns:
        ``index = x.index``、``columns = [lower, median, upper]`` の DataFrame。

    Raises:
        ValueError: ``models`` の件数が 3 でない場合。
    """
    sorted_alphas = sorted(models.keys())
    if len(sorted_alphas) != _EXPECTED_NUM_QUANTILES:
        raise ValueError(
            f"3 つの α が必要です（low / median / high）: "
            f"got {len(sorted_alphas)} 件 {sorted_alphas}"
        )

    lower_alpha, median_alpha, upper_alpha = sorted_alphas
    lower = np.asarray(models[lower_alpha].predict(x))
    median = np.asarray(models[median_alpha].predict(x))
    upper = np.asarray(models[upper_alpha].predict(x))

    if return_yen:
        lower = np.exp(lower)
        median = np.exp(median)
        upper = np.exp(upper)

    return pd.DataFrame(
        {LOWER_COL: lower, MEDIAN_COL: median, UPPER_COL: upper},
        index=x.index,
    )


def load_quantile_models(model_dir: Path) -> dict[float, LGBMRegressor]:
    """``models/lgbm_quantile_{low,med,high}.pkl`` を一括ロードする.

    Args:
        model_dir: モデルディレクトリ。

    Returns:
        ``{alpha: モデル}`` の辞書。α はモデルの ``alpha`` 属性から復元する。

    Raises:
        FileNotFoundError: いずれかのモデルファイルが存在しない場合。
    """
    suffixes = ("low", "med", "high")
    models: dict[float, LGBMRegressor] = {}
    for suffix in suffixes:
        path = model_dir / f"lgbm_quantile_{suffix}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"分位点モデルが見つかりません: {path}")
        model: LGBMRegressor = joblib.load(path)
        alpha = float(model.alpha)
        models[alpha] = model
        logger.info(f"モデル読込: {path} (α={alpha})")
    return models


def empirical_interval_from_similar(
    df: pd.DataFrame,
    target_idx: int,
    n_similar: int = 30,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    price_column: str = "取引価格（総額）",
) -> dict[str, float]:
    """類似物件の実取引価格の分位点から予測区間を出す.

    Returns:
        {lower, median, upper, n_used} の辞書
    """
    # Phase 1 の関数で類似 N 件を取得
    similar = find_similar_properties(df, target_idx, n_neighbors=n_similar)
    prices = similar[price_column].astype(float)

    return {
        "lower": float(prices.quantile(lower_q)),
        "median": float(prices.quantile(0.50)),
        "upper": float(prices.quantile(upper_q)),
        "n_used": len(prices),
    }


def compare_intervals(
    quantile_interval: dict[str, float],  # Phase 2の出力
    empirical_interval: dict[str, float],  # Phase 3の出力
    divergence_threshold: float = 0.30,  # 中央値が30%以上ずれたら警告
) -> dict[str, Any]:
    """2 つの区間を比較し、乖離度と信頼度フラグを返す.

    Returns:
        {
          divergence: float,        # 中央値の相対乖離
          is_reliable: bool,        # 乖離が閾値以内か
          flag: str,                # "高信頼" / "要人手査定"
          ...
        }
    """
    q_med = quantile_interval[MEDIAN_COL]
    e_med = empirical_interval[MEDIAN_COL]

    # ゼロ除算ガード（実証中央値が 0 になることは実質ないが念のため）
    if e_med <= 0:
        raise ValueError(f"実証中央値が不正です: {e_med}")

    # 中央値の相対乖離（実証側を基準にする = 事実を分母に置く）
    divergence = abs(q_med - e_med) / e_med
    is_reliable = divergence <= divergence_threshold

    # 区間の重なり（overlap）も見ると、より精密な判定ができる
    overlap_lower = max(quantile_interval[LOWER_COL], empirical_interval[LOWER_COL])
    overlap_upper = min(quantile_interval[UPPER_COL], empirical_interval[UPPER_COL])
    has_overlap = overlap_lower <= overlap_upper

    return {
        DIVERGENCE_COL: divergence,
        "is_reliable": is_reliable,
        "has_overlap": has_overlap,
        "flag": "高信頼" if is_reliable and has_overlap else "要人手査定",
        "quantile_median": q_med,
        "empirical_median": e_med,
        "n_similar_used": empirical_interval.get("n_used"),
    }


def divergence_direction(ratio: float) -> str:
    """倍率（ratio = quantile/empirical）から方向ラベルを返す.

    Returns:
        ``"高値"``（モデルが実取引より高い）/ ``"安値"`` / ``"一致"``。
    """
    if ratio > 1.0:
        return "高値"
    if ratio < 1.0:
        return "安値"
    return "一致"


def divergence_band(ratio: float) -> str:
    """倍率から重大度バンド（軽度 / 中度 / 重度）を返す.

    高値側（ratio>1）と安値側（ratio<1）を対称に扱うため、
    ``fold = max(ratio, 1/ratio)``（常に 1.0 以上の「離れ具合」）で評価する。

    Args:
        ratio: ``quantile_median / empirical_median``（正の値）。

    Raises:
        ValueError: ``ratio`` が 0 以下の場合。
    """
    if ratio <= 0:
        raise ValueError(f"ratio は正の値である必要があります: {ratio}")
    fold = max(ratio, 1.0 / ratio)
    for boundary, label in _BAND_BOUNDARIES:
        if fold < boundary:
            return label
    return _SEVERE_BAND


def assess_reliability(
    comparison: dict[str, Any],
    n_used: int,
    min_similar: int = 10,
) -> dict[str, Any]:
    """査定の信頼度を 高 / 中 / 低 の 3 段階で判定する.

    Phase 4（Streamlit 物件査定ページ）の信頼度バッジ用ロジック。
    Quantile 区間と実証的区間の重なり・乖離バンド・類似物件数を総合して、
    顧客に提示する信頼度ラベルとメッセージを返す。

    判定基準:
        - 低: 2 区間が重ならない、または乖離が重度（モデルが分布外を外挿）
        - 中: 乖離が中度、または類似物件数が ``min_similar`` 未満（実証側が不安定）
        - 高: 区間が重なり、乖離が軽度で、類似物件も十分

    Args:
        comparison: ``compare_intervals`` の戻り値（``quantile_median`` /
            ``empirical_median`` / ``has_overlap`` を含む）。
        n_used: 実証的区間に使った類似物件数。
        min_similar: これ未満なら実証側が不安定とみなす類似物件数の下限。

    Returns:
        ``level``（高/中/低）・``band``・``direction``・``ratio``・``message`` を持つ辞書。
    """
    ratio = comparison["quantile_median"] / comparison["empirical_median"]
    band = divergence_band(ratio)
    direction = divergence_direction(ratio)
    has_overlap = comparison["has_overlap"]

    if not has_overlap or band == _SEVERE_BAND:
        level = "低"
        message = "モデル予測が実取引と大きく乖離。人手査定を推奨します。"
    elif band == "中度" or n_used < min_similar:
        level = "中"
        message = "乖離または類似物件数に留意。参考値として扱ってください。"
    else:
        level = "高"
        message = "モデル予測と実取引が整合しています。"

    return {
        "level": level,
        "band": band,
        "direction": direction,
        "ratio": ratio,
        "message": message,
    }


def add_interpretability_columns(table: pd.DataFrame) -> pd.DataFrame:
    """乖離度テーブルに解釈しやすい列を加える.

    ``quantile_median`` / ``empirical_median`` から、倍率・符号付き乖離率・
    方向・重大度バンドを算出する。``divergence``（絶対値）より直感的に
    「どちらへどれだけ外れているか」を読めるようにするための列群。

    Args:
        table: ``quantile_median`` と ``empirical_median`` 列を持つ DataFrame。

    Returns:
        ``ratio``・``signed_pct``・``direction``・``band`` を追加した新しい DataFrame。
        空テーブルの場合は当該列を空で付与して返す。
    """
    out = table.copy()
    if out.empty:
        for col in (RATIO_COL, "signed_pct", "direction", "band"):
            out[col] = pd.Series(dtype="object" if col in ("direction", "band") else "float64")
        return out

    q = out["quantile_median"].astype(float)
    e = out["empirical_median"].astype(float)
    out[RATIO_COL] = q / e
    out["signed_pct"] = (q - e) / e * 100.0  # 符号付き: + ならモデルが高い
    out["direction"] = out[RATIO_COL].map(divergence_direction)
    out["band"] = out[RATIO_COL].map(divergence_band)
    return out


def build_divergence_table(
    df: pd.DataFrame,
    models: dict[float, LGBMRegressor],
    model_features: Sequence[str],
    n_similar: int = 30,
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    id_columns: Sequence[str] = (),
    return_yen: bool = True,
) -> pd.DataFrame:
    """全物件について Quantile中央値 と 実証中央値 の乖離度テーブルを作る.

    Quantile Regression（Phase 2）の予測中央値と、類似物件の実取引価格から
    求めた中央値（Phase 3）の相対乖離を物件ごとに算出する。乖離が大きい物件は
    「モデルが学習データの分布外を外挿している」サインとして人手査定の対象になる。

    Args:
        df: 全取引データ（``model_features`` と類似物件検索に必要な列を含む）。
        models: ``{alpha: 学習済み LGBMRegressor}`` の辞書。3 件必要。
        model_features: モデルが期待する特徴量列の順序付きリスト。
        n_similar: 実証的区間に使う類似物件数。
        divergence_threshold: ``flag`` 判定に使う乖離度の閾値。
        id_columns: 出力に含める識別用の列（市区町村コード・面積など）。
        return_yen: Quantile 予測を円換算するか（実証中央値は常に円）。

    Returns:
        ``index``・``id_columns``・``divergence``・``quantile_median``・
        ``empirical_median``・``has_overlap``・``n_used``・``flag`` を持つ DataFrame。
        実証中央値が 0 以下の行（類似物件が存在しない等）は除外する。
    """
    intervals = predict_with_interval(models, df[list(model_features)], return_yen=return_yen)

    records: list[dict[str, Any]] = []
    skipped = 0
    for idx in df.index:
        empirical = empirical_interval_from_similar(df, idx, n_similar=n_similar)
        # 実証中央値が 0 以下なら比較不能なのでスキップ（compare_intervals は raise する）
        if empirical[MEDIAN_COL] <= 0:
            skipped += 1
            continue

        quantile = {
            LOWER_COL: float(intervals.at[idx, LOWER_COL]),
            MEDIAN_COL: float(intervals.at[idx, MEDIAN_COL]),
            UPPER_COL: float(intervals.at[idx, UPPER_COL]),
        }
        comparison = compare_intervals(quantile, empirical, divergence_threshold)

        records.append(
            {
                "index": idx,
                **{col: df.at[idx, col] for col in id_columns},
                DIVERGENCE_COL: comparison[DIVERGENCE_COL],
                "quantile_median": comparison["quantile_median"],
                "empirical_median": comparison["empirical_median"],
                "has_overlap": comparison["has_overlap"],
                "n_used": empirical["n_used"],
                "flag": comparison["flag"],
            }
        )

    logger.info(f"乖離度テーブル作成: {len(records)} 件（スキップ {skipped} 件）")
    return add_interpretability_columns(pd.DataFrame.from_records(records))


def filter_high_divergence(
    table: pd.DataFrame,
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """乖離度テーブルから閾値を超える物件を乖離度の降順で抽出する.

    Args:
        table: ``build_divergence_table`` が返すテーブル。
        threshold: この値を超える物件を「高乖離」として抽出する。

    Returns:
        ``divergence > threshold`` の行を乖離度降順に並べた DataFrame。
    """
    high = table[table[DIVERGENCE_COL] > threshold]
    return high.sort_values(DIVERGENCE_COL, ascending=False).reset_index(drop=True)


# 市区町村集計の既定列名・集計キー
COUNT_COL = "件数"
RATIO_MEDIAN_COL = "ratio_median"
OVER_RATE_COL = "over_rate"
NAME_COL = "市区町村名"
_MUNICIPALITY_CODE_COL = "市区町村コード"

# 集計の並び替えモード（high: ratio 降順 / low: ratio 昇順）
_VALID_MODES = ("high", "low")


def aggregate_divergence_by_municipality(
    table: pd.DataFrame,
    mode: str = "high",
    municipality_column: str = _MUNICIPALITY_CODE_COL,
    ratio_column: str = RATIO_COL,
    min_count: int = 10,
    name_by_code: dict[int, str] | None = None,
) -> pd.DataFrame:
    """乖離度テーブルを市区町村ごとに集計し、高乖離 / 低乖離の傾向順に並べる.

    各市区町村について件数・``ratio`` 中央値・モデル高値割合（``ratio > 1`` の比率）を
    集計する。``ratio = quantile_median / empirical_median`` なので、中央値が大きい
    市区町村ほど「モデルが実取引より高値を出しがち」と読める。

    Args:
        table: ``ratio`` 列と市区町村コード列を持つ DataFrame
            （``build_divergence_table`` の出力や、その CSV 読み込み結果を想定）。
        mode: ``"high"`` なら ``ratio`` 中央値の降順（モデル高値の地域を上位に）、
            ``"low"`` なら昇順（モデル安値の地域を上位に）。
        municipality_column: 集計キーとなる市区町村コード列名。
        ratio_column: 倍率列名。
        min_count: この件数未満の市区町村は除外する（少数によるノイズを抑える）。
        name_by_code: 市区町村コード → 名称の辞書。渡すと ``市区町村名`` 列を付与する。

    Returns:
        市区町村ごとの集計テーブル（``市区町村コード`` / ``件数`` /
        ``ratio_median`` / ``over_rate`` と、辞書を渡した場合は ``市区町村名``）。
        ``mode`` に応じて整列済み。

    Raises:
        ValueError: ``mode`` が ``"high"`` / ``"low"`` 以外の場合。
        KeyError: 必須列（市区町村コード / ratio）が ``table`` に無い場合。
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode は {_VALID_MODES} のいずれかです: got {mode!r}")

    missing = [c for c in (municipality_column, ratio_column) if c not in table.columns]
    if missing:
        raise KeyError(f"集計に必要な列がありません: {missing}")

    agg = (
        table.groupby(municipality_column)
        .agg(
            **{
                COUNT_COL: (ratio_column, "size"),
                RATIO_MEDIAN_COL: (ratio_column, "median"),
                OVER_RATE_COL: (ratio_column, lambda s: float((s > 1.0).mean() * 100)),
            }
        )
        .reset_index()
    )

    agg = agg[agg[COUNT_COL] >= min_count]
    # mode="low" のときだけ昇順（安値方向を上位に）
    agg = agg.sort_values(RATIO_MEDIAN_COL, ascending=(mode == "low")).reset_index(drop=True)

    if name_by_code is not None:
        agg[NAME_COL] = agg[municipality_column].map(lambda c: name_by_code.get(int(c), str(c)))

    return agg
