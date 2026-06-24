"""アプリ全体で共有する可視化スタイル定数とヘルパー.

ページ間で配色・グラフの余白・高さを揃え、UI の一貫性を保つための単一ソース。
各ページはここから定数を import して使う。
"""

from typing import Any

# 誤差系の連続カラースケール（低=青 → 高=赤）。精度・誤差を表す図で共通利用する。
ERROR_COLOR_SCALE = "RdYlBu_r"

# カテゴリ系（価格・件数など誤差でない量）の基調色。
BRAND_COLOR = "#185FA5"

# 区間較正ゲージの色（達成=緑 / 未達=アンバー）。
GAUGE_OK_COLOR = "#1D9E75"
GAUGE_WARN_COLOR = "#E0A030"

# グラフ共通の高さ・余白。サマリ用のコンパクトな図に合わせる。
CHART_HEIGHT = 260
CHART_MARGIN = {"l": 10, "r": 10, "t": 30, "b": 10}


def apply_chart_style(fig: Any) -> Any:
    """Plotly 図に共通の高さ・余白を適用して返す.

    Args:
        fig: Plotly の Figure。

    Returns:
        スタイル適用済みの同一 Figure。
    """
    fig.update_layout(height=CHART_HEIGHT, margin=CHART_MARGIN)
    return fig
