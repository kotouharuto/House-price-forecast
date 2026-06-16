"""BIダッシュボード: 予測結果をグラフで多面的に可視化するページ.

``outputs/test_predictions.csv`` を読み込み、KPIサマリ・予測精度グラフ・
エリア別ランキングを、サイドバーのフィルタと連動して表示する。
地図可視化は別ページ（後続フェーズ）で提供する。
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# プロジェクト内モジュール（sys.path 操作後である必要があるため E402 を許容）
from app.filters import render_sidebar_filters  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    ACTUAL_PRICE_COL,
    AGE_COL,
    APE_COL,
    AREA_COL,
    ERROR_YEN_COL,
    PRED_PRICE_COL,
    PRICE_BAND_COL,
    STATION_COL,
    WARD_CODE_COL,
    aggregate_by_station,
    aggregate_by_ward,
    load_predictions,
    summarize_metrics,
    worst_properties,
)
from src.visualization.format import format_yen_jp  # noqa: E402
from src.visualization.master import code_to_label, load_municipality_names  # noqa: E402

# 散布図の最大プロット点数（描画負荷を抑えるためサンプリングする）
_MAX_SCATTER_POINTS = 4000
# 誤差を表す連続カラースケール（低=青 → 高=赤）
_ERROR_COLOR_SCALE = "RdYlBu_r"

st.set_page_config(page_title="BIダッシュボード", layout="wide")


@st.cache_data
def _load() -> pd.DataFrame:
    """予測結果を読み込む（キャッシュ付き）."""
    return load_predictions()


@st.cache_data
def _municipality_names() -> dict[int, str]:
    """市区町村コード→名称マスタを読み込む（キャッシュ付き）."""
    return load_municipality_names()


@st.cache_data
def _aggregate_ward_cached(df: pd.DataFrame) -> pd.DataFrame:
    """行政区別の集計(キャッシュ付き)"""
    return aggregate_by_ward(df)


@st.cache_data
def _aggregate_station_cached(df: pd.DataFrame) -> pd.DataFrame:
    """駅別の集計"""
    return aggregate_by_station(df)


@st.cache_data
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """DataFrameを Excel 互換の UTF-8 BOM 付き CSV バイト列に変換する（キャッシュ付き）."""
    return df.to_csv(index=False).encode("utf-8-sig")


def _render_kpis(df: pd.DataFrame) -> None:
    """KPIサマリ（件数・R²・MAE・RMSE・MAPE・Median APE）をカード表示する."""
    metrics = summarize_metrics(df)
    st.subheader("KPIサマリ")

    cols = st.columns(6)
    cols[0].metric("件数", f"{int(metrics['count']):,}", help="表示対象の物件件数")
    cols[1].metric(
        "R²(log)",
        f"{metrics['r2_log']:.3f}",
        help="log価格スケールでの決定係数。1に近いほど予測が良い",
    )
    cols[2].metric("MAE", format_yen_jp(metrics["mae_yen"]), help="平均絶対誤差（円）")
    cols[3].metric(
        "RMSE",
        format_yen_jp(metrics["rmse_yen"]),
        help="二乗平均平方根誤差（円）。大きな外れに敏感",
    )
    cols[4].metric("MAPE", f"{metrics['mape']:.1f}%", help="平均絶対パーセント誤差（%）")
    cols[5].metric(
        "Median APE",
        f"{metrics['median_ape']:.1f}%",
        help="絶対パーセント誤差の中央値（%）。外れ値に頑健",
    )
    st.caption("各指標の意味は docs/modeling/evaluation_metrics.md を参照。")


def _sample_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """散布図用に点数を上限までサンプリングする（再現性のため固定シード）."""
    if len(df) <= _MAX_SCATTER_POINTS:
        return df
    return df.sample(n=_MAX_SCATTER_POINTS, random_state=42)


def _render_sampling_note(total: int, shown: int) -> None:
    """サンプリング表示時に、全件中の表示件数を注記する."""
    if shown < total:
        st.caption(
            f"※ 描画負荷軽減のため、全{total:,}件中{shown:,}件をサンプリング表示しています。"
        )


def _render_scatter(df: pd.DataFrame) -> None:
    """予測 vs 実測の散布図（y=x基準線つき、色=APE）を描画する."""
    st.subheader("予測 vs 実測")
    plot_df = _sample_for_plot(df).copy()
    # ホバーに億・万表記を出すための補助列（軸は円のまま、補助としてツールチップに表示）
    plot_df["_actual_jp"] = plot_df[ACTUAL_PRICE_COL].map(format_yen_jp)
    plot_df["_pred_jp"] = plot_df[PRED_PRICE_COL].map(format_yen_jp)

    fig = px.scatter(
        plot_df,
        x=ACTUAL_PRICE_COL,
        y=PRED_PRICE_COL,
        color=APE_COL,
        color_continuous_scale=_ERROR_COLOR_SCALE,
        opacity=0.5,
        custom_data=["_actual_jp", "_pred_jp"],
        labels={
            ACTUAL_PRICE_COL: "実測価格(円)",
            PRED_PRICE_COL: "予測価格(円)",
            APE_COL: "APE(%)",
        },
    )
    # y=x基準線を追加する前に散布点のみホバーを書き換える
    fig.update_traces(
        hovertemplate=(
            "実測価格: %{customdata[0]}<br>"
            "予測価格: %{customdata[1]}<br>"
            "APE: %{marker.color:.1f}%<extra></extra>"
        )
    )
    limit = float(max(plot_df[ACTUAL_PRICE_COL].max(), plot_df[PRED_PRICE_COL].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="y=x（完全予測）",
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_sampling_note(len(df), len(plot_df))


def _render_error_distribution(df: pd.DataFrame) -> None:
    """誤差分布のヒストグラム（APE / 残差を切替）を描画する."""
    st.subheader("誤差分布")
    metric = st.radio("表示する指標", ["APE(%)", "残差(円)"], horizontal=True, key="errdist")

    if metric == "APE(%)":
        fig = px.histogram(df, x=APE_COL, nbins=50, labels={APE_COL: "APE(%)"})
    else:
        # 残差はCSVの error_yen（= 予測 - 実測）をそのまま使い、符号定義を一本化する
        fig = px.histogram(
            df, x=ERROR_YEN_COL, nbins=50, labels={ERROR_YEN_COL: "残差(円) = 予測 - 実測"}
        )
    st.plotly_chart(fig, use_container_width=True)


def _render_price_band(df: pd.DataFrame) -> None:
    """価格帯別の件数（棒）と平均APE（色）を描画する."""
    st.subheader("価格帯別の件数と精度")
    grouped = (
        df.groupby(PRICE_BAND_COL, observed=True)
        .agg(
            count=(PRED_PRICE_COL, "size"),
            mean_ape=(APE_COL, "mean"),
            mean_actual=(ACTUAL_PRICE_COL, "mean"),
        )
        .reset_index()
        .sort_values("mean_actual")  # 価格の安い帯から並べる
    )
    fig = px.bar(
        grouped,
        x=PRICE_BAND_COL,
        y="count",
        color="mean_ape",
        color_continuous_scale=_ERROR_COLOR_SCALE,
        labels={PRICE_BAND_COL: "価格帯", "count": "件数", "mean_ape": "平均APE(%)"},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_ranking(df: pd.DataFrame, *, by_station: bool) -> None:
    """エリア別（行政区 or 駅）のランキング棒グラフを描画する."""
    if by_station:
        st.subheader("駅別ランキング")
        agg = _aggregate_station_cached(df)
        key_col = STATION_COL
    else:
        st.subheader("行政区別ランキング")
        agg = _aggregate_ward_cached(df)
        key_col = WARD_CODE_COL

    widget_key = "station_rank" if by_station else "ward_rank"
    metric_label = st.selectbox("ランキング指標", ["平均予測価格", "平均APE"], key=widget_key)
    value_col, ascending = (
        ("pred_price_mean", False) if metric_label == "平均予測価格" else ("mape", True)
    )

    top = agg.sort_values(value_col, ascending=ascending).head(20).copy()
    if by_station:
        top["label"] = top[key_col].astype(str)
    else:
        # 行政区は市区町村名でラベル表示（未知コードはコード文字列にフォールバック）
        name_by_code = _municipality_names()
        top["label"] = top[key_col].map(lambda code: code_to_label(code, name_by_code))

    # 棒のラベル: 予測価格は億・万、APEは%で表示する
    if value_col == "pred_price_mean":
        top["_bar_text"] = top[value_col].map(format_yen_jp)
    else:
        top["_bar_text"] = top[value_col].map(lambda v: f"{v:.1f}%")

    fig = px.bar(
        top,
        x=value_col,
        y="label",
        orientation="h",
        text="_bar_text",
        color="mape",  # 棒の色=平均APE。精度の良し悪しを直感的に示す
        color_continuous_scale=_ERROR_COLOR_SCALE,
        labels={value_col: metric_label, "label": "エリア", "mape": "平均APE(%)"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def _render_feature_relation(df: pd.DataFrame) -> None:
    """選択した特徴量（面積 or 築年数）× 予測価格の散布図を描画する.

    X軸に選ばなかった方の特徴量を色に割り当て、2つの特徴量を同時に俯瞰できる。
    """
    st.subheader("特徴量と価格の関係")

    # X軸は日本語ラベルで選ばせ、選択値から列名を逆引きする
    feature_labels = {AREA_COL: "面積(㎡)", AGE_COL: "築年数"}
    label_to_col = {label: col for col, label in feature_labels.items()}
    x_label = st.selectbox("X軸の特徴量", list(feature_labels.values()), key="feature_x")
    x_col = label_to_col[x_label]
    # X軸に選ばれなかった方の特徴量を色に割り当てる
    color_col = AGE_COL if x_col == AREA_COL else AREA_COL

    plot_df = _sample_for_plot(df)
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=PRED_PRICE_COL,
        color=color_col,
        opacity=0.5,
        labels={**feature_labels, PRED_PRICE_COL: "予測価格(円)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_sampling_note(len(df), len(plot_df))


def _render_worst_properties(df: pd.DataFrame) -> None:
    """残差が大きい順のワースト物件テーブル（N-3）を描画する."""
    st.subheader("ワースト物件")
    col_sort, col_n = st.columns([1, 1])
    sort_label = col_sort.selectbox(
        "ソート基準",
        ["残差(絶対値)", "APE"],
        key="worst_sort",
    )
    n = col_n.slider("表示件数", min_value=10, max_value=200, value=50, step=10, key="worst_n")

    sort_by = "abs_error" if sort_label == "残差(絶対値)" else "ape"
    table = worst_properties(df, sort_by=sort_by, n=n).copy()

    # 行政区コードを名称に置換して可読性を上げる
    name_by_code = _municipality_names()
    table["市区町村コード"] = table["市区町村コード"].map(
        lambda code: code_to_label(code, name_by_code)
    )
    table = table.rename(columns={"市区町村コード": "行政区"})

    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_download_section(
    filtered: pd.DataFrame, ward_summary: pd.DataFrame, station_summary: pd.DataFrame
) -> None:
    """フィルタ後データ・行政区集計・駅集計をCSVダウンロード可能にする（N-2）."""
    with st.expander("ダウンロード"):
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "フィルタ後データ(CSV)",
            data=_to_csv_bytes(filtered),
            file_name="filtered_predictions.csv",
            mime="text/csv",
        )
        col2.download_button(
            "行政区集計(CSV)",
            data=_to_csv_bytes(ward_summary),
            file_name="ward_summary.csv",
            mime="text/csv",
        )
        col3.download_button(
            "駅集計(CSV)",
            data=_to_csv_bytes(station_summary),
            file_name="station_summary.csv",
            mime="text/csv",
        )


def main() -> None:
    """ページのエントリポイント."""
    st.title("予測結果 BIダッシュボード")

    df = _load()
    filtered = render_sidebar_filters(df, _municipality_names())
    st.caption(f"表示対象: {len(filtered):,} 件 / 全 {len(df):,} 件")

    if filtered.empty:
        st.warning("条件に一致する物件がありません。フィルタを緩めてください。")
        return

    _render_kpis(filtered)
    _render_download_section(
        filtered, _aggregate_ward_cached(filtered), _aggregate_station_cached(filtered)
    )

    tab_accuracy, tab_area, tab_feature, tab_worst = st.tabs(
        ["予測精度", "エリア別", "特徴量", "ワースト物件"]
    )
    with tab_accuracy:
        _render_scatter(filtered)
        _render_error_distribution(filtered)
        _render_price_band(filtered)
    with tab_area:
        _render_ranking(filtered, by_station=False)
        _render_ranking(filtered, by_station=True)
    with tab_feature:
        _render_feature_relation(filtered)
    with tab_worst:
        _render_worst_properties(filtered)


main()
