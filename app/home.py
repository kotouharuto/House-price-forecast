"""東京都不動産価格予測 BIツールのトップページ.

プロジェクト概要・データソース・モデルの説明と、現行モデルの主要評価指標サマリ、
および各ページ（BIダッシュボード／地図／評価指標一覧）への導線を提供する。
"""

import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# プロジェクトルートを sys.path に追加（src.xxx / app.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.assets import download_assets  # noqa: E402
from app.theme import BRAND_COLOR, ERROR_COLOR_SCALE, apply_chart_style  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    DEFAULT_NOMINAL_COVERAGE,
    PRICE_BAND_COL,
    WARD_CODE_COL,
    aggregate_by_ward,
    coverage_rate,
    interval_width_stats,
    load_predictions,
    metrics_by_price_band,
    percent_error_rates,
    price_band_order,
    summarize_metrics,
)
from src.visualization.format import format_yen_jp  # noqa: E402
from src.visualization.master import code_to_label, load_municipality_names  # noqa: E402

st.set_page_config(
    page_title="東京都不動産価格予測 BIツール",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def _load() -> pd.DataFrame:
    """予測結果を読み込む（キャッシュ付き）."""
    return load_predictions()


@st.cache_data
def _municipality_names() -> dict[int, str]:
    """市区町村コード→名称マスタを読み込む（キャッシュ付き）."""
    return load_municipality_names()


def _render_overview() -> None:
    """プロジェクト概要・データソース・モデル説明を表示する."""
    st.title("東京都不動産価格予測 BIツール")
    st.markdown(
        """
        国土交通省「不動産情報ライブラリ」のオープンデータを学習し、
        LightGBM 回帰モデルで東京都の不動産取引価格を予測する End-to-End プロジェクトです。
        本アプリでは、予測結果を多面的に可視化・評価できます。
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("データソース")
        st.markdown(
            """
            - **出典**: 国土交通省「不動産情報ライブラリ」（オープンデータ）
            - **対象エリア**: 東京都
            - **主な特徴量**: 種類・面積・築年数・最寄駅・駅徒歩・構造・用途地域 等
            - **目的変数**: `取引価格（総額）`（log 変換して学習）
            """
        )
    with col2:
        st.subheader("モデル")
        st.markdown(
            """
            - **アルゴリズム**: LightGBM（Huber 損失、外れ値耐性つき）
            - **設定管理**: `configs/model_params.yaml`
            - **学習ログ**: `logs/train.log`
            - **予測結果**: `outputs/test_predictions.csv`
            """
        )


def _render_kpi_summary(df: pd.DataFrame) -> None:
    """主要評価指標のサマリカードを表示する."""
    metrics = summarize_metrics(df)
    pe = percent_error_rates(df)

    st.subheader("現行モデルの主要評価指標")
    st.caption("test セット全件で算出した値。詳細は『評価指標一覧』ページを参照してください。")

    row1 = st.columns(4)
    row1[0].metric("件数", f"{int(metrics['count']):,}", help="評価対象の物件数")
    row1[1].metric(
        "R²(log)",
        f"{metrics['r2_log']:.3f}",
        help="log 価格スケールでの決定係数。1 に近いほど良い",
    )
    row1[2].metric(
        "MAE",
        format_yen_jp(metrics["mae_yen"]),
        help="平均絶対誤差（円）。「平均で何円ずれるか」の指標",
    )
    row1[3].metric(
        "RMSE",
        format_yen_jp(metrics["rmse_yen"]),
        help="円スケールの二乗平均平方根誤差。大きな外れ予測に敏感",
    )

    row2 = st.columns(4)
    row2[0].metric(
        "MAPE",
        f"{metrics['mape']:.2f}%",
        help="平均絶対パーセント誤差。比率誤差の平均",
    )
    row2[1].metric(
        "Median APE",
        f"{metrics['median_ape']:.2f}%",
        help="絶対パーセント誤差の中央値。外れ値に頑健",
    )
    row2[2].metric(
        "PE10",
        f"{pe['PE10']:.1f}%",
        help="APE が 10% 以下の物件比率（業界 AVM の Good 帯目安: 50% 以上）",
    )
    row2[3].metric(
        "PE20",
        f"{pe['PE20']:.1f}%",
        help="APE が 20% 以下の物件比率（業界 AVM の Good 帯目安: 80% 以上）",
    )


def _render_band_summary_table(df: pd.DataFrame) -> None:
    """価格帯別の Median APE / PE10 のサマリを表形式で表示する（詳細用）."""
    band_df = metrics_by_price_band(df)
    if band_df.empty:
        return

    display = band_df.rename(
        columns={
            "actual_price_band": "価格帯",
            "count": "件数",
            "median_ape": "Median APE (%)",
            "mape": "MAPE (%)",
            "pe10": "PE10 (%)",
            "pe20": "PE20 (%)",
        }
    )
    # 表示用に四捨五入
    for col in ("Median APE (%)", "MAPE (%)", "PE10 (%)", "PE20 (%)"):
        display[col] = display[col].round(2)

    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_accuracy_chart(df: pd.DataFrame) -> None:
    """価格帯別の Median APE を横棒グラフで表示する（低いほど良い）."""
    band = metrics_by_price_band(df)
    if band.empty:
        st.info("価格帯別の精度を計算できませんでした。")
        return

    order = price_band_order(df)
    band[PRICE_BAND_COL] = pd.Categorical(band[PRICE_BAND_COL], categories=order, ordered=True)
    band = band.sort_values(PRICE_BAND_COL)

    fig = px.bar(
        band,
        x="median_ape",
        y=PRICE_BAND_COL,
        orientation="h",
        color="median_ape",
        color_continuous_scale=ERROR_COLOR_SCALE,
        text=band["median_ape"].map(lambda v: f"{v:.1f}%"),
        labels={"median_ape": "Median APE(%)", PRICE_BAND_COL: "価格帯"},
    )
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": order},
        coloraxis_showscale=False,
    )
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_interval_gauge(df: pd.DataFrame) -> None:
    """予測区間のカバレッジ（PICP）を目標90%ラインつきゲージで表示する."""
    picp = coverage_rate(df)
    if math.isnan(picp):
        st.info("予測区間の列がありません。`predict_test` で再生成してください。")
        return

    widths = interval_width_stats(df)
    gap = picp - DEFAULT_NOMINAL_COVERAGE
    fill = "#1D9E75" if gap >= -2 else "#E0A030"
    gap_color = "#0F6E56" if gap >= -2 else "#C0392B"
    piaw = format_yen_jp(widths["width_median_yen"])

    st.markdown(
        f"""
        <div style="border:1px solid #e6e9ef;border-radius:12px;padding:16px 18px;">
          <div style="display:flex;align-items:baseline;gap:8px;">
            <span style="font-size:30px;font-weight:600;">{picp:.1f}%</span>
            <span style="font-size:13px;color:{gap_color};">{gap:+.1f}pt vs 目標</span>
          </div>
          <div style="position:relative;margin:14px 0 6px;height:16px;
                      background:#eef0f2;border-radius:8px;">
            <div style="position:absolute;left:0;top:0;bottom:0;width:{picp:.1f}%;
                        background:{fill};border-radius:8px;"></div>
            <div style="position:absolute;left:{DEFAULT_NOMINAL_COVERAGE:.0f}%;top:-4px;bottom:-4px;
                        width:2px;background:#31333F;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#8b8f9a;">
            <span>0%</span><span>目標 {DEFAULT_NOMINAL_COVERAGE:.0f}%</span><span>100%</span>
          </div>
          <div style="font-size:13px;color:#6b7280;margin-top:10px;">
            区間幅中央値(PIAW): {piaw}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_area_highlight(df: pd.DataFrame) -> None:
    """平均予測価格の上位行政区を横棒グラフで表示する."""
    ward = aggregate_by_ward(df)
    if ward.empty or "pred_price_mean" not in ward.columns:
        return

    name_by_code = _municipality_names()
    top = ward.sort_values("pred_price_mean", ascending=False).head(8).copy()
    top["label"] = top[WARD_CODE_COL].map(lambda code: code_to_label(code, name_by_code))
    top["_text"] = top["pred_price_mean"].map(format_yen_jp)

    fig = px.bar(
        top,
        x="pred_price_mean",
        y="label",
        orientation="h",
        text="_text",
        labels={"pred_price_mean": "平均予測価格(円)", "label": "行政区"},
    )
    fig.update_traces(marker_color=BRAND_COLOR)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_analysis_summary(df: pd.DataFrame) -> None:
    """分析・可視化のサマリ帯（精度・区間較正・エリア）を表示する."""
    st.subheader("分析サマリ")
    st.caption("ひと目で全体像を把握できるサマリ。詳細は各ページへ。")

    col_acc, col_picp = st.columns(2)
    with col_acc:
        st.markdown("##### 価格帯別の精度（Median APE）")
        _render_accuracy_chart(df)
        st.caption("詳しくは『BIダッシュボード』")
    with col_picp:
        st.markdown("##### 予測区間の較正（PICP）")
        _render_interval_gauge(df)
        st.caption("詳しくは『評価指標一覧』")

    st.markdown("##### エリアハイライト（平均予測価格 上位）")
    _render_area_highlight(df)
    st.caption("詳しくは『地図』")

    with st.expander("価格帯別の詳細（表）"):
        _render_band_summary_table(df)


def _render_navigation() -> None:
    """各ページへの導線をボタン状に表示する."""
    st.subheader("各ページへの導線")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            """
            ### BIダッシュボード
            KPI・散布図・誤差分布・ランキング・特徴量・ワースト物件などを多面的に確認。
            CSV ダウンロードも可能。
            """
        )
    with cols[1]:
        st.markdown(
            """
            ### 地図
            行政区単位のコロプレス、駅単位の円マーカー。
            粒度トグルで切替、クリックで詳細ポップアップ。
            """
        )
    with cols[2]:
        st.markdown(
            """
            ### 評価指標一覧
            各指標の定義・計算式・読み方・業界基準との比較。
            """
        )
    with cols[3]:
        st.markdown(
            """
            ### 物件査定
            点予測＋予測区間＋類似物件の実取引区間を統合表示。
            査定信頼度を高/中/低で判定。
            """
        )
    st.info("左サイドバーの『Pages』からそれぞれのページへ移動できます。")


def main() -> None:
    """エントリポイント."""
    # Streamlit Cloud では HuggingFace Hub からアセットを事前取得（ローカルなら即帰る）
    download_assets()

    _render_overview()
    st.divider()

    try:
        df = _load()
    except FileNotFoundError as err:
        st.error(f"予測結果CSVが見つかりません: {err}")
        st.info(
            "`uv run python -m src.modeling.train` でモデルを学習し、"
            "`outputs/test_predictions.csv` を生成してください。"
        )
        return

    _render_kpi_summary(df)
    st.divider()
    _render_analysis_summary(df)
    st.divider()
    _render_navigation()


main()
