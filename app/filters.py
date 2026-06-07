"""サイドバーのフィルタUI（BIページ・地図ページ共通）.

予測結果DataFrameを受け取り、サイドバーにフィルタUIを構築して絞り込み後の
DataFrameを返す純粋なUIヘルパー。各ページで重複していたフィルタ実装を一本化する。
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.visualization.aggregate import (  # noqa: E402
    AGE_COL,
    AREA_COL,
    TYPE_COL,
    WARD_CODE_COL,
    available_stations,
    filter_predictions,
    price_band_order,
)
from src.visualization.master import code_to_label  # noqa: E402


def render_sidebar_filters(df: pd.DataFrame, name_by_code: dict[int, str]) -> pd.DataFrame:
    """サイドバーのフィルタUIを構築し、絞り込み後のDataFrameを返す.

    選択肢は全件データから決めるため、フィルタ操作によらず選択肢は安定する。
    最寄駅は選択中の行政区に連動して候補を絞り込む。

    Args:
        df: フィルタ前の全予測結果。
        name_by_code: 市区町村コード→名称の対応辞書。

    Returns:
        フィルタ適用後のデータフレーム。
    """
    st.sidebar.header("フィルタ")

    # 行政区は利便性のため市区町村名で選択させ、内部ではコードに変換して絞り込む
    present_codes = sorted(df[WARD_CODE_COL].dropna().unique().tolist())
    ward_labels = [code_to_label(code, name_by_code) for code in present_codes]
    label_to_code = {code_to_label(code, name_by_code): int(code) for code in present_codes}
    # ウィジェットには明示キーを付与してページ間で選択状態を共有する
    # （選択肢は常に全件データから決めるため、フィルタ操作で options から外れる心配は無い）
    selected_ward_labels = st.sidebar.multiselect(
        "行政区（市区町村名）", ward_labels, key="flt_wards"
    )
    selected_wards = [label_to_code[label] for label in selected_ward_labels]

    # 最寄駅は選択中の行政区内の駅のみに連動して絞り込む（未選択時は全駅）
    station_options = available_stations(df, selected_wards)
    selected_stations = st.sidebar.multiselect("最寄駅", station_options, key="flt_stations")

    types = sorted(df[TYPE_COL].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect("物件種類", types, key="flt_types")

    # 価格帯は金額順（平均実測価格の昇順）で選択肢を並べる
    bands = price_band_order(df)
    selected_bands = st.sidebar.multiselect("価格帯", bands, key="flt_bands")

    area_min, area_max = float(df[AREA_COL].min()), float(df[AREA_COL].max())
    area_range = st.sidebar.slider(
        "面積（㎡）", area_min, area_max, (area_min, area_max), key="flt_area"
    )

    age_min, age_max = float(df[AGE_COL].min()), float(df[AGE_COL].max())
    age_range = st.sidebar.slider("築年数", age_min, age_max, (age_min, age_max), key="flt_age")

    yamanote_label = st.sidebar.radio(
        "山手線内側", ["すべて", "内側のみ", "外側のみ"], key="flt_yamanote"
    )
    yamanote_map: dict[str, bool | None] = {
        "すべて": None,
        "内側のみ": True,
        "外側のみ": False,
    }

    return filter_predictions(
        df,
        ward_codes=selected_wards,
        stations=selected_stations,
        property_types=selected_types,
        price_bands=selected_bands,
        area_range=area_range,
        age_range=age_range,
        yamanote_inside=yamanote_map[yamanote_label],
    )
