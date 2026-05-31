"""地図ダッシュボード: 駅単位/行政区単位の予測結果を地図で可視化するページ.

サイドバーのフィルタと連動して、選択した粒度（行政区 or 駅）で集計結果を
地図に描画する。行政区はGeoJSONによるコロプレス、駅は円マーカー。
クリックで当該エリアの詳細をポップアップ表示する。初期表示は行政区単位。
"""

import html
import sys
from pathlib import Path
from typing import Any

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
from branca.element import Element
from streamlit_folium import st_folium

# プロジェクトルートを sys.path に追加（src.xxx / app.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.filters import render_sidebar_filters  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    STATION_COL,
    WARD_CODE_COL,
    load_predictions,
    station_map_summary,
    ward_map_summary,
)
from src.visualization.format import format_yen_jp  # noqa: E402
from src.visualization.geo import DEFAULT_CODE_PROPERTY, load_municipality_geojson  # noqa: E402
from src.visualization.master import code_to_label, load_municipality_names  # noqa: E402

# 東京の地図中心とズーム
_TOKYO_CENTER = (35.68, 139.76)
_INITIAL_ZOOM = 11
# 駅マーカー用の連続カラースケール（低=青 → 中=黄 → 高=赤）
_STATION_COLORS = ["#2c7bb6", "#ffffbf", "#d7191c"]
# 行政区コロプレス用の連続カラースケール（folium組み込みパレット）
_WARD_FILL_COLOR = "YlOrRd"
_WARD_LEGEND_COLORS = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]

# 色分け指標: ラベル -> (集計列, 金額表示か)
_METRIC_OPTIONS: dict[str, tuple[str, bool]] = {
    "平均予測価格": ("pred_price_mean", True),
    "平均実測価格": ("actual_price_mean", True),
    "平均APE": ("mape", False),
}

# 粒度トグルの選択肢（仕様: 初期表示=行政区）
_GRANULARITY_OPTIONS = ["行政区", "駅"]

st.set_page_config(page_title="地図", layout="wide")


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


@st.cache_data
def _ward_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    """行政区単位の地図用集計（キャッシュ付き）."""
    return ward_map_summary(df)


@st.cache_data(show_spinner=False)
def _geojson_cached() -> dict[str, Any]:
    """東京都市区町村のGeoJSONを読み込む（キャッシュ付き）."""
    return load_municipality_geojson()


def _or_dash(value: object) -> str:
    """欠損値はダッシュ表示にフォールバックする."""
    return "—" if pd.isna(value) else str(value)


def _format_legend_value(value: float, *, is_money: bool) -> str:
    """地図凡例に収まる短めの値表記を返す."""
    if is_money:
        return format_yen_jp(value)
    return f"{value:.1f}%"


def _add_compact_legend(
    fmap: folium.Map,
    values: pd.Series,
    *,
    colors: list[str],
    metric_label: str,
    is_money: bool,
) -> None:
    """Foliumの過密な自動凡例の代わりに、最小/最大だけの凡例を追加する."""
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return

    vmin = float(numeric_values.min())
    vmax = float(numeric_values.max())
    min_label = html.escape(_format_legend_value(vmin, is_money=is_money))
    max_label = html.escape(_format_legend_value(vmax, is_money=is_money))
    title = html.escape(f"{metric_label}（{'円' if is_money else '%'}）")
    gradient = ", ".join(colors)

    legend_html = f"""
    <style>
      .trp-map-legend {{
        position: absolute;
        right: 16px;
        bottom: 24px;
        z-index: 9999;
        width: min(280px, calc(100% - 32px));
        padding: 10px 12px;
        border: 1px solid rgba(31, 41, 55, 0.18);
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 2px 8px rgba(31, 41, 55, 0.16);
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 12px;
        line-height: 1.35;
      }}
      .trp-map-legend__title {{
        margin-bottom: 6px;
        font-weight: 600;
      }}
      .trp-map-legend__bar {{
        height: 10px;
        border-radius: 999px;
        background: linear-gradient(to right, {gradient});
      }}
      .trp-map-legend__labels {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-top: 5px;
        white-space: nowrap;
      }}
    </style>
    <div class="trp-map-legend">
      <div class="trp-map-legend__title">{title}</div>
      <div class="trp-map-legend__bar"></div>
      <div class="trp-map-legend__labels">
        <span>{min_label}</span>
        <span>{max_label}</span>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))


# ---------- 駅ブランチ（円マーカー） ----------


def _popup_html_station(row: pd.Series) -> str:
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


def _build_station_map(
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
    if vmax <= vmin:  # 駅が1件等で範囲が潰れる場合のゼロ除算回避
        vmax = vmin + 1.0
    colormap = cm.LinearColormap(colors=_STATION_COLORS, vmin=vmin, vmax=vmax)
    _add_compact_legend(
        fmap,
        stations[metric_col],
        colors=_STATION_COLORS,
        metric_label=metric_label,
        is_money=is_money,
    )

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
            popup=folium.Popup(_popup_html_station(row), max_width=300),
        ).add_to(fmap)
    return fmap


# ---------- 行政区ブランチ（コロプレス） ----------


def _format_money_or_dash(value: object) -> str:
    """金額をフォーマット、欠損はダッシュにする."""
    if pd.isna(value):
        return "—"
    return format_yen_jp(float(value))  # type: ignore[arg-type]


def _format_percent_or_dash(value: object) -> str:
    """%をフォーマット、欠損はダッシュにする."""
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f}%"  # type: ignore[arg-type]


def _enrich_geojson_with_metrics(
    geojson_data: dict[str, Any],
    summary: pd.DataFrame,
    name_by_code: dict[int, str],
    code_property: str,
) -> dict[str, Any]:
    """GeoJSONの ``feature.properties`` へポップアップ表示用の整形済み文字列を埋め込む.

    元のGeoJSONを破壊しないよう、各featureはコピーしてから属性を追加する。
    """
    metrics_by_code = summary.set_index(WARD_CODE_COL).to_dict("index")
    enriched: list[dict[str, Any]] = []
    for feat in geojson_data.get("features", []):
        new_feat = {**feat, "properties": {**feat.get("properties", {})}}
        try:
            code = int(str(new_feat["properties"][code_property]))
        except (KeyError, ValueError, TypeError):
            continue
        metrics = metrics_by_code.get(code, {})
        if metrics:
            new_feat["properties"].update(
                {
                    "ward_name": code_to_label(code, name_by_code),
                    "count_text": f"{int(metrics['count']):,} 件",
                    "pred_text": (
                        f"平均 {_format_money_or_dash(metrics.get('pred_price_mean'))} / "
                        f"中央 {_format_money_or_dash(metrics.get('pred_price_median'))}"
                    ),
                    "actual_text": (
                        f"平均 {_format_money_or_dash(metrics.get('actual_price_mean'))} / "
                        f"中央 {_format_money_or_dash(metrics.get('actual_price_median'))}"
                    ),
                    "ape_text": (
                        f"平均 {_format_percent_or_dash(metrics.get('mape'))} / "
                        f"中央 {_format_percent_or_dash(metrics.get('median_ape'))}"
                    ),
                    "repr_text": (
                        f"{_or_dash(metrics.get('repr_type'))} / "
                        f"{_or_dash(metrics.get('repr_band'))}"
                    ),
                }
            )
        else:
            new_feat["properties"].update(
                {
                    "ward_name": code_to_label(code, name_by_code),
                    "count_text": "—",
                    "pred_text": "—",
                    "actual_text": "—",
                    "ape_text": "—",
                    "repr_text": "—",
                }
            )
        enriched.append(new_feat)
    return {**geojson_data, "features": enriched}


def _build_ward_map(
    summary: pd.DataFrame,
    metric_col: str,
    *,
    is_money: bool,
    metric_label: str,
    geojson_data: dict[str, Any],
    name_by_code: dict[int, str],
) -> folium.Map:
    """行政区集計＋GeoJSONからコロプレス地図を構築する."""
    fmap = folium.Map(
        location=list(_TOKYO_CENTER), zoom_start=_INITIAL_ZOOM, tiles="cartodbpositron"
    )
    code_property = DEFAULT_CODE_PROPERTY
    enriched_geojson = _enrich_geojson_with_metrics(
        geojson_data, summary, name_by_code, code_property
    )

    # Choropleth はキー列を文字列で比較するため、市区町村コードを str に変換しておく
    choro_df = summary[[WARD_CODE_COL, metric_col]].copy()
    choro_df[WARD_CODE_COL] = choro_df[WARD_CODE_COL].astype(str)
    choropleth = folium.Choropleth(
        geo_data=enriched_geojson,
        data=choro_df,
        columns=[WARD_CODE_COL, metric_col],
        key_on=f"feature.properties.{code_property}",
        fill_color=_WARD_FILL_COLOR,
        fill_opacity=0.7,
        line_opacity=0.3,
        nan_fill_color="lightgray",
        nan_fill_opacity=0.3,
    )
    if choropleth.color_scale is not None:
        choropleth._children.pop(choropleth.color_scale.get_name(), None)
        choropleth.color_scale = None
    choropleth.add_to(fmap)
    _add_compact_legend(
        fmap,
        summary[metric_col],
        colors=_WARD_LEGEND_COLORS,
        metric_label=metric_label,
        is_money=is_money,
    )

    # 透明レイヤを重ねて、クリック時のツールチップ/ポップアップを担当させる（M-5）
    folium.GeoJson(
        enriched_geojson,
        name="行政区情報",
        style_function=lambda _: {"fillOpacity": 0.0, "color": "transparent", "weight": 0},
        highlight_function=lambda _: {"fillOpacity": 0.2, "fillColor": "#000000"},
        # ラベルなしツールチップにして、クリック取得時に行政区名がそのまま返るようにする
        tooltip=folium.GeoJsonTooltip(
            fields=["ward_name"], aliases=[""], labels=False, sticky=False
        ),
        popup=folium.GeoJsonPopup(
            fields=["ward_name", "count_text", "pred_text", "actual_text", "ape_text", "repr_text"],
            aliases=["行政区", "件数", "予測価格", "実測価格", "APE", "代表 種類/価格帯"],
            max_width=320,
        ),
    ).add_to(fmap)

    return fmap


# ---------- メイン ----------

# 地図クリック→フィルタ連動（N-1）で使う session_state のキー
_LAST_CLICK_MARKER_KEY = "_map_last_applied_click"
_PENDING_FILTER_UPDATE_KEY = "_map_pending_filter_update"


def _apply_pending_map_filter_update() -> None:
    """地図クリックで予約したフィルタ更新を、ウィジェット生成前に反映する."""
    pending_update = st.session_state.pop(_PENDING_FILTER_UPDATE_KEY, None)
    if not pending_update:
        return
    for key, value in pending_update.items():
        st.session_state[key] = value


def _apply_map_click_to_filters(granularity: str, last_clicked_tooltip: str | None) -> bool:
    """地図クリックの内容をサイドバーフィルタへ書き戻す（N-1 / 置換セマンティクス）.

    Args:
        granularity: ``"行政区"`` または ``"駅"``。
        last_clicked_tooltip: ``st_folium`` が返す ``last_object_clicked_tooltip``。
            行政区は GeoJsonTooltip で行政区名がそのまま入る前提。
            駅は ``"<駅名>（<指標>）"`` 形式から駅名を切り出す。

    Returns:
        フィルタを実際に更新した場合 ``True``、未更新なら ``False``。
    """
    if not last_clicked_tooltip:
        return False
    marker = (granularity, last_clicked_tooltip)
    # 同じクリックを何度も適用しない（リランループ防止）
    if st.session_state.get(_LAST_CLICK_MARKER_KEY) == marker:
        return False
    st.session_state[_LAST_CLICK_MARKER_KEY] = marker

    if granularity == "行政区":
        # クリックされた区のみをフィルタに置換し、駅フィルタは整合のためクリアする
        ward_name = last_clicked_tooltip.strip()
        st.session_state[_PENDING_FILTER_UPDATE_KEY] = {
            "flt_wards": [ward_name],
            "flt_stations": [],
        }
    else:
        # 駅マーカーのツールチップは "<駅名>（<指標>）" 形式。右側を剥がして駅名を得る
        station_name = last_clicked_tooltip.rsplit("（", 1)[0].strip()
        st.session_state[_PENDING_FILTER_UPDATE_KEY] = {"flt_stations": [station_name]}
    return True


def _clear_map_click_selection() -> None:
    """『クリック選択をクリア』ボタンの処理: 行政区/駅フィルタとクリックマーカーを消す."""
    for key in ("flt_wards", "flt_stations", _LAST_CLICK_MARKER_KEY, _PENDING_FILTER_UPDATE_KEY):
        st.session_state.pop(key, None)


def main() -> None:
    """ページのエントリポイント."""
    st.title("予測結果 地図")

    df = _load()
    name_by_code = _municipality_names()
    _apply_pending_map_filter_update()
    filtered = render_sidebar_filters(df, name_by_code)
    if filtered.empty:
        st.warning("条件に一致する物件がありません。フィルタを緩めてください。")
        return

    # 仕様: 初期表示=行政区
    col_g, col_m, col_clear = st.columns([2, 2, 1])
    granularity = col_g.radio(
        "粒度", _GRANULARITY_OPTIONS, horizontal=True, index=0, key="map_granularity"
    )
    metric_label = col_m.selectbox("色分け指標", list(_METRIC_OPTIONS), key="map_metric")
    metric_col, is_money = _METRIC_OPTIONS[metric_label]
    # 地図クリックで設定した行政区/駅フィルタを解除
    col_clear.button(
        "クリック選択をクリア",
        on_click=_clear_map_click_selection,
        help="地図クリックで自動設定した行政区/駅フィルタをまとめて解除します",
    )

    if granularity == "行政区":
        try:
            geojson_data = _geojson_cached()
        except FileNotFoundError as err:
            st.error(f"行政区GeoJSONが見つかりません: {err}")
            st.info(
                "`configs/tokyo_municipalities.geojson` に国土数値情報 N03 のGeoJSONを"
                "配置するか、粒度を「駅」に切り替えてください。"
            )
            return

        summary = _ward_summary_cached(filtered)
        if summary.empty:
            st.warning("該当する行政区がありません。")
            return
        st.caption(f"表示対象: {len(filtered):,} 件 / 行政区 {len(summary):,} 区")
        fmap = _build_ward_map(
            summary,
            metric_col,
            is_money=is_money,
            metric_label=metric_label,
            geojson_data=geojson_data,
            name_by_code=name_by_code,
        )
    else:  # 駅
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
        fmap = _build_station_map(plotted, metric_col, is_money=is_money, metric_label=metric_label)

    # クリック内容だけを返してもらい、それ以外（ズーム等）では再描画させない
    result = st_folium(
        fmap,
        use_container_width=True,
        height=600,
        returned_objects=["last_object_clicked_tooltip"],
    )
    last_tooltip = result.get("last_object_clicked_tooltip") if isinstance(result, dict) else None
    if _apply_map_click_to_filters(granularity, last_tooltip):
        # 既に上で描画したサイドバーは古い値のため、再実行して新フィルタを反映する
        st.rerun()


main()
