"""東京都不動産価格予測 BIツールのトップページ.

プロジェクト概要・データソース・モデルの説明と、現行モデルの主要評価指標サマリ、
および各ページ（BIダッシュボード／地図／評価指標一覧）への導線を提供する。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# プロジェクトルートを sys.path に追加（src.xxx / app.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.visualization.aggregate import (  # noqa: E402
    load_predictions,
    metrics_by_price_band,
    percent_error_rates,
    summarize_metrics,
)
from src.visualization.format import format_yen_jp  # noqa: E402

st.set_page_config(
    page_title="東京都不動産価格予測 BIツール",
    page_icon="🏠",
    layout="wide",
)

# ---------- Streamlit Cloud 用のアセット自動ダウンロード ----------
# HuggingFace Hub に置いた学習済みモデル・予測結果ファイルを、ローカルに無ければ取得する。
# 既存ファイルがある（ローカル開発）場合はスキップ。
_HF_REPO_ID = "Haruto0321/tokyo-rent-predictor-data"
_DOWNLOAD_FILES: list[tuple[str, Path]] = [
    ("models/lgbm_model.pkl", _PROJECT_ROOT / "models" / "lgbm_model.pkl"),
    ("outputs/test_predictions.csv", _PROJECT_ROOT / "outputs" / "test_predictions.csv"),
    (
        "outputs/test_predictions_properties.geojson",
        _PROJECT_ROOT / "outputs" / "test_predictions_properties.geojson",
    ),
    (
        "outputs/test_predictions_stations.geojson",
        _PROJECT_ROOT / "outputs" / "test_predictions_stations.geojson",
    ),
]


@st.cache_resource(show_spinner="データを準備しています...")
def _download_assets() -> None:
    """HuggingFace Hub から必要なファイルをダウンロードする.

    既にローカルに存在する場合はスキップする（ローカル開発環境への配慮）。
    """
    import shutil

    from huggingface_hub import hf_hub_download

    for hf_path, local_path in _DOWNLOAD_FILES:
        if local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(_PROJECT_ROOT),
        )
        # hf_hub_download はキャッシュディレクトリに置くため、指定パスへコピー
        if Path(downloaded) != local_path:
            shutil.copy2(downloaded, local_path)


@st.cache_data
def _load() -> pd.DataFrame:
    """予測結果を読み込む（キャッシュ付き）."""
    return load_predictions()


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


def _render_band_summary(df: pd.DataFrame) -> None:
    """価格帯別の Median APE / PE10 のサマリを表形式で表示する."""
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

    st.subheader("価格帯別の精度")
    st.caption(
        "価格帯ごとの誤差水準。Median APE が低く・PE10 が高いほど精度が良い。"
        "両端（低額帯・高額帯）に難があるかを把握する用途。"
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_navigation() -> None:
    """各ページへの導線をボタン状に表示する."""
    st.subheader("各ページへの導線")
    cols = st.columns(3)
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
    st.info("左サイドバーの『Pages』からそれぞれのページへ移動できます。")


def main() -> None:
    """エントリポイント."""
    # Streamlit Cloud では HuggingFace Hub からアセットを事前取得（ローカルなら即帰る）
    _download_assets()

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
    _render_band_summary(df)
    st.divider()
    _render_navigation()


main()
