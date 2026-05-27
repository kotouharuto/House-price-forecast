"""地図ダッシュボード（駅単位）: 駅ごとの予測結果を地図マーカーで可視化するページ.

``outputs/test_predictions.csv`` を読み込み、サイドバーのフィルタと連動して、
駅単位の集計値（予測価格・実測価格・APE）を地図上の円マーカーで表示する。
マーカークリックで当該駅の詳細をポップアップ表示する。行政区単位は後続フェーズ。
"""

import sys
from pathlib import Path

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# プロジェクトルートを sys.path に追加（src.xxx / app.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.filters import render_sidebar_filters  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    STATION_COL,
    load_predictions,
    station_map_summary,
)
from src.visualization.format import format_yen_jp  # noqa: E402
from src.visualization.master import load_municipality_names  # noqa: E402

# 東京の地図中心とズーム
_TOKYO_CENTER = (35.68, 139.76)
_INITIAL_ZOOM = 11
# 誤差カラースケール（低=青 → 中=黄 → 高=赤）。グラフ側と方向を合わせる
_COLORS = ["#2c7bb6", "#ffffbf", "#d7191c"]

# 色分け指標: ラベル -> (集計列, 金額表示か)
_METRIC_OPTIONS: dict[str, tuple[str, bool]] = {
    "平均予測価格": ("pred_price_mean", True),
    "平均実測価格": ("actual_price_mean", True),
    "平均APE": ("mape", False),
}

st.set_page_config(page_title="地図（駅単位）", layout="wide")


@st.cache_data
def _load() -> pd.DataFrame:
    """予測結果を読み込む（キャッシュ付き）."""
    return load_predictions()


@st.cache_data
def _municipality_names() -> dict[int, str]:
    """市区町村コード→名称マスタを読み込む（キャッシュ付き）."""
    return load_municipality_names()


@st.cache_data
def _station_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    """駅単位の地図用集計（キャッシュ付き）."""
    return station_map_summary(df)


def _or_dash(value: object) -> str:
    """欠損値はダッシュ表示にフォールバックする."""
    return "—" if pd.isna(value) else str(value)


def _popup_html(row: pd.Series) -> str:
    """駅マーカーのポップアップHTMLを組み立てる."""
    return (
        f"<b>{row[STATION_COL]}</b><br>"
        f"物件件数: {int(row['count']):,} 件<br>"
        f"予測価格 平均: {format_yen_jp(row['pred_price_mean'])} / "
        f"中央: {format_yen_jp(row['pred_price_median'])}<br>"
        f"実測価格 平均: {format_yen_jp(row['actual_price_mean'])} / "
        f"中央: {format_yen_jp(row['actual_price_median'])}<br>"
        f"平均APE: {row['mape']:.1f}% / Median APE: {row['median_ape']:.1f}%<br>"
        f"代表種類: {_or_dash(row['repr_type'])} / 代表価格帯: {_or_dash(row['repr_band'])}"
    )


def _build_map(
    stations: pd.DataFrame, metric_col: str, *, is_money: bool, metric_label: str
) -> folium.Map:
    """駅集計から円マーカーつきの地図を構築する.

    マーカーの色は選択指標、半径は物件件数（平方根スケール）で表す。
    """
    fmap = folium.Map(
        location=list(_TOKYO_CENTER), zoom_start=_INITIAL_ZOOM, tiles="cartodbpositron"
    )

    vmin = float(stations[metric_col].min())
    vmax = float(stations[metric_col].max())
    if vmax <= vmin:  # 駅が1件等で範囲が潰れる場合のゼロ除算を回避
        vmax = vmin + 1.0
    colormap = cm.LinearColormap(colors=_COLORS, vmin=vmin, vmax=vmax)
    colormap.caption = f"{metric_label}（{'円' if is_money else '%'}）"
    colormap.add_to(fmap)

    max_count = int(stations["count"].max())
    for _, row in stations.iterrows():
        value = float(row[metric_col])
        # 件数を半径(5〜18px)に平方根スケール
        radius = 5 + 13 * (int(row["count"]) / max_count) ** 0.5
        marker_color = colormap(value)
        label = format_yen_jp(value) if is_money else f"{value:.1f}%"
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=radius,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7,
            weight=1,
            tooltip=f"{row[STATION_COL]}（{label}）",
            popup=folium.Popup(_popup_html(row), max_width=300),
        ).add_to(fmap)
    return fmap


def main() -> None:
    """ページのエントリポイント."""
    st.title("予測結果 地図（駅単位）")

    df = _load()
    filtered = render_sidebar_filters(df, _municipality_names())
    if filtered.empty:
        st.warning("条件に一致する物件がありません。フィルタを緩めてください。")
        return

    summary = _station_summary_cached(filtered)
    plotted = summary.dropna(subset=["lat", "lon"])
    excluded = len(summary) - len(plotted)

    note = f"表示対象: {len(filtered):,} 件 / 地図表示 {len(plotted):,} 駅"
    if excluded:
        note += f"（緯度経度欠損の {excluded:,} 駅は地図から除外）"
    st.caption(note)

    if plotted.empty:
        st.warning("地図に表示できる駅がありません（緯度経度がすべて欠損）。")
        return

    metric_label = st.selectbox("色分け指標", list(_METRIC_OPTIONS))
    metric_col, is_money = _METRIC_OPTIONS[metric_label]

    fmap = _build_map(plotted, metric_col, is_money=is_money, metric_label=metric_label)
    st_folium(fmap, use_container_width=True, height=600, returned_objects=[])


main()
