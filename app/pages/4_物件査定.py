"""物件査定ページ（説明可能性ロードマップ Phase 4）.

対象物件を1件選び、点予測・モデル予測区間（Phase 2）・類似物件からの実証的区間
（Phase 3）を1画面に統合して提示する。2区間の重なりと乖離から査定信頼度を
3段階（高/中/低）で判定し、顧客説明に耐える「誠実な不確実性表示」を行う。
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# プロジェクトルートを sys.path に追加（src.xxx / app.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.appraisal import (  # noqa: E402
    PRICE_COL,
    apply_range,
    build_candidate_pool,
    property_label,
)
from app.assets import download_assets  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    AGE_COL,
    AREA_COL,
    STATION_DISTANCE_COL,
    TYPE_COL,
    WARD_CODE_COL,
)
from src.visualization.export_high_divergence import (  # noqa: E402
    extract_model_features,
    load_features,
)
from src.visualization.format import format_yen_jp  # noqa: E402
from src.visualization.master import code_to_label, load_municipality_names  # noqa: E402
from src.visualization.prediction import (  # noqa: E402
    LOWER_COL,
    MEDIAN_COL,
    UPPER_COL,
    assess_reliability,
    compare_intervals,
    empirical_interval_from_similar,
    load_quantile_models,
    predict_with_interval,
)
from src.visualization.similar_properties import find_similar_properties  # noqa: E402

# 取引価格列のエイリアス（ページ内の既存参照を維持するため）
_PRICE_COL = PRICE_COL
_MODEL_DIR = _PROJECT_ROOT / "models"
_POINT_MODEL_PATH = _MODEL_DIR / "lgbm_model.pkl"

# 実証的区間に使う類似物件数（査定の母集団サイズ）
_N_SIMILAR = 30

# 信頼度レベル → Streamlit のステータス表示関数のマッピング
_LEVEL_RENDERERS = {"高": st.success, "中": st.warning, "低": st.error}

st.set_page_config(page_title="物件査定", page_icon="🏠", layout="wide")


@st.cache_data(show_spinner="取引データを読み込んでいます...")
def _load_data() -> pd.DataFrame:
    """特徴量データ（category 変換済み）を読み込む."""
    return load_features()


@st.cache_resource(show_spinner="モデルを読み込んでいます...")
def _load_models() -> tuple[object, dict[float, object], list[str]]:
    """点予測モデル・分位点モデル・モデル特徴量を読み込む."""
    point_model = joblib.load(_POINT_MODEL_PATH)
    quantile_models = load_quantile_models(_MODEL_DIR)
    model_features = extract_model_features(quantile_models)
    return point_model, quantile_models, model_features


@st.cache_data
def _load_name_by_code() -> dict[int, str]:
    """市区町村コード→名称の辞書を読み込む."""
    return load_municipality_names()


def _range_filter(
    candidates: pd.DataFrame,
    column: str,
    label: str,
    key: str,
    unit_div: float = 1.0,
) -> pd.DataFrame:
    """指定列の範囲スライダーをサイドバーに出し、絞り込み後の候補を返す.

    候補内の最小・最大を初期範囲とする。値が1種類しかない（min==max）場合は
    スライダーを出さずにそのまま返す。``unit_div`` は表示単位の変換用
    （価格を万円表示にする等）。絞り込みの実体は ``apply_range`` に委譲する。

    Args:
        candidates: 絞り込み対象の候補 DataFrame。
        column: 絞り込みに使う数値列。
        label: スライダーのラベル。
        key: ウィジェットの一意キー。
        unit_div: 表示値へ変換する除数（例: 価格は 1e4 で万円表示）。

    Returns:
        範囲条件を満たす候補 DataFrame。
    """
    valid = candidates[column].dropna()
    if valid.empty:
        return candidates

    lo, hi = float(valid.min()) / unit_div, float(valid.max()) / unit_div
    if lo >= hi:
        return candidates

    selected = st.sidebar.slider(label, lo, hi, (lo, hi), key=key)
    return apply_range(candidates, column, selected[0] * unit_div, selected[1] * unit_div)


def _select_target(df: pd.DataFrame, name_by_code: dict[int, str]) -> int | None:
    """サイドバーで対象物件を絞り込んで選択し、df 上の index を返す.

    行政区 → 種類 で母集団を絞ったうえで、占有面積・築年数・駅徒歩・価格を
    各項目ごとに範囲指定して候補を絞り込み、個別物件を選ぶ。該当物件が無ければ
    ``None`` を返す。
    """
    st.sidebar.header("査定対象の選択")

    ward_codes = sorted(df[WARD_CODE_COL].dropna().unique().tolist())
    ward_labels = {code_to_label(c, name_by_code): int(c) for c in ward_codes}
    ward_label = st.sidebar.selectbox("行政区", list(ward_labels.keys()))
    ward_code = ward_labels[ward_label]

    ward_df = df[df[WARD_CODE_COL] == ward_code]
    types = sorted(ward_df[TYPE_COL].dropna().unique().tolist())
    if not types:
        return None
    ptype = st.sidebar.selectbox("種類", types)

    candidates = build_candidate_pool(df, ward_code, ptype)
    if candidates.empty:
        return None

    # 各項目ごとの範囲条件で候補を絞り込む
    st.sidebar.subheader("条件で絞り込み")
    candidates = _range_filter(candidates, AREA_COL, "占有面積（㎡）", key="flt_apr_area")
    candidates = _range_filter(candidates, AGE_COL, "築年数", key="flt_apr_age")
    candidates = _range_filter(
        candidates, STATION_DISTANCE_COL, "駅徒歩（分）", key="flt_apr_station"
    )
    candidates = _range_filter(
        candidates, _PRICE_COL, "価格（万円）", key="flt_apr_price", unit_div=1e4
    )

    candidates = candidates.sort_values(_PRICE_COL)
    if candidates.empty:
        return None

    st.sidebar.caption(f"該当 {len(candidates)} 件")
    target_idx = st.sidebar.selectbox(
        "物件",
        options=list(candidates.index),
        format_func=lambda i: property_label(df.loc[i]),
    )
    return int(target_idx)


def _appraise(
    df: pd.DataFrame,
    target_idx: int,
    point_model: object,
    quantile_models: dict[float, object],
    model_features: list[str],
) -> dict:
    """対象物件の点予測・2区間・信頼度を計算して辞書で返す."""
    x = df.loc[[target_idx], model_features]

    # 点予測（log スケールで学習しているため exp で円に戻す）
    point_log = float(point_model.predict(x)[0])
    point_yen = float(np.exp(point_log))

    # Phase 2: モデル予測区間
    interval = predict_with_interval(quantile_models, x).iloc[0]
    quantile = {
        LOWER_COL: float(interval[LOWER_COL]),
        MEDIAN_COL: float(interval[MEDIAN_COL]),
        UPPER_COL: float(interval[UPPER_COL]),
    }

    # Phase 3: 類似物件からの実証的区間
    empirical = empirical_interval_from_similar(df, target_idx, n_similar=_N_SIMILAR)

    comparison = compare_intervals(quantile, empirical)
    reliability = assess_reliability(comparison, n_used=int(empirical["n_used"]))

    return {
        "point_yen": point_yen,
        "quantile": quantile,
        "empirical": empirical,
        "comparison": comparison,
        "reliability": reliability,
    }


def _build_interval_figure(result: dict) -> go.Figure:
    """モデル予測区間と実証的区間を同一価格軸に並べた図を作る."""
    quantile = result["quantile"]
    empirical = result["empirical"]
    point_yen = result["point_yen"]

    fig = go.Figure()
    rows = (
        ("モデル予測", quantile, "#85B7EB", "#185FA5"),
        ("実取引(類似)", empirical, "#5DCAA5", "#0F6E56"),
    )
    for label, interval, bar_color, marker_color in rows:
        fig.add_trace(
            go.Scatter(
                x=[interval[LOWER_COL], interval[UPPER_COL]],
                y=[label, label],
                mode="lines",
                line={"color": bar_color, "width": 18},
                showlegend=False,
                hovertemplate="%{x:,.0f}円<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[interval[MEDIAN_COL]],
                y=[label],
                mode="markers",
                marker={"color": marker_color, "size": 16, "symbol": "line-ns-open"},
                showlegend=False,
                hovertemplate="中央値 %{x:,.0f}円<extra></extra>",
            )
        )

    fig.add_vline(
        x=point_yen,
        line_dash="dash",
        line_color="#888780",
        annotation_text="点予測",
        annotation_position="top",
    )
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="価格（円）",
        yaxis={"categoryorder": "array", "categoryarray": ["実取引(類似)", "モデル予測"]},
    )
    return fig


def _render_result(df: pd.DataFrame, target_idx: int, result: dict) -> None:
    """査定結果（バッジ・メトリクス・区間図・類似物件）を描画する."""
    reliability = result["reliability"]
    quantile = result["quantile"]
    empirical = result["empirical"]

    target = df.loc[target_idx]
    st.subheader(f"査定結果: {property_label(target)}")

    # 信頼度バッジ（高/中/低でステータス色を変える）
    renderer = _LEVEL_RENDERERS[reliability["level"]]
    renderer(
        f"査定信頼度: {reliability['level']}（{reliability['message']}）"
        f" / モデルは実取引より {reliability['direction']}・{reliability['band']}"
    )

    # メトリクスカード
    cols = st.columns(4)
    cols[0].metric("点予測", format_yen_jp(result["point_yen"]))
    cols[1].metric(
        "モデル予測区間",
        f"{format_yen_jp(quantile[LOWER_COL])} 〜 {format_yen_jp(quantile[UPPER_COL])}",
    )
    cols[2].metric("乖離 (ratio)", f"{reliability['ratio']:.2f}倍")
    cols[3].metric("類似物件数", f"{int(empirical['n_used'])}件")

    # 2区間の比較図
    st.plotly_chart(_build_interval_figure(result), use_container_width=True)

    # 類似物件テーブルと価格分布
    similar = find_similar_properties(df, target_idx, n_neighbors=_N_SIMILAR)
    left, right = st.columns(2)
    with left:
        st.markdown("##### 類似物件 上位5件")
        display = similar.head(5)[[AREA_COL, AGE_COL, STATION_DISTANCE_COL, _PRICE_COL]].rename(
            columns={_PRICE_COL: "取引価格"}
        )
        display["取引価格"] = display["取引価格"].map(format_yen_jp)
        st.dataframe(display, use_container_width=True, hide_index=True)
    with right:
        st.markdown("##### 類似物件の価格分布")
        hist = go.Figure(go.Histogram(x=similar[_PRICE_COL], nbinsx=20, marker_color="#9FE1CB"))
        hist.add_vline(x=result["point_yen"], line_dash="dash", line_color="#185FA5")
        hist.update_layout(
            height=300,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title="取引価格（円）",
            yaxis_title="件数",
        )
        st.plotly_chart(hist, use_container_width=True)


def main() -> None:
    """エントリポイント."""
    st.title("物件査定")
    st.caption(
        "対象物件の点予測に加え、モデル予測区間（Phase 2）と類似物件の実取引区間"
        "（Phase 3）を並べて表示し、査定の信頼度を判定します。"
    )

    # Streamlit Cloud では HuggingFace Hub からアセットを事前取得（ローカルなら即帰る）。
    # ページ直リンクで開かれても home.py を経由せず取得できるよう、本ページでも呼ぶ。
    download_assets()

    try:
        df = _load_data()
        point_model, quantile_models, model_features = _load_models()
        name_by_code = _load_name_by_code()
    except FileNotFoundError as err:
        st.error(f"必要なファイルが見つかりません: {err}")
        st.info(
            "ローカルで `uv run python -m src.modeling.train` と "
            "`uv run python -m src.modeling.train_quantile` を実行し、"
            "モデルと特徴量データを生成してください。"
        )
        return

    target_idx = _select_target(df, name_by_code)
    if target_idx is None:
        st.warning("該当する物件がありません。サイドバーの条件を変更してください。")
        return

    try:
        result = _appraise(df, target_idx, point_model, quantile_models, model_features)
    except ValueError as err:
        # 類似物件が母集団に存在しない等
        st.error(f"査定できませんでした: {err}")
        return

    _render_result(df, target_idx, result)


main()
