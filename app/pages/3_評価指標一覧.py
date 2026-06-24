"""評価指標一覧ページ.

予測モデルで利用する各評価指標の定義・計算式・解釈・読み方の早見表に加え、
現行モデルの数値と業界 AVM 基準（NAVAR/IAAO）との比較を表示する。
"""

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.visualization.aggregate import (  # noqa: E402
    DEFAULT_NOMINAL_COVERAGE,
    coverage_rate,
    interval_width_stats,
    load_predictions,
    metrics_by_price_band,
    percent_error_rates,
    summarize_metrics,
)
from src.visualization.format import format_yen_jp  # noqa: E402

st.set_page_config(page_title="評価指標一覧", page_icon="📐", layout="wide")


@st.cache_data
def _load() -> pd.DataFrame:
    """予測結果を読み込む（キャッシュ付き）."""
    return load_predictions()


# 業界 AVM（NAVAR/IAAO 等）の評価グレード基準
_GRADE_BENCHMARKS = pd.DataFrame(
    [
        ["Excellent", "< 5%", "> 70%", "> 90%"],
        ["Good", "5〜10%", "50〜70%", "80〜90%"],
        ["Acceptable", "10〜15%", "40〜50%", "70〜80%"],
        ["Marginal", "15〜20%", "30〜40%", "50〜70%"],
        ["Poor", "> 20%", "< 30%", "< 50%"],
    ],
    columns=["グレード", "Median APE", "PE10", "PE20"],
)


def _render_current_summary(df: pd.DataFrame) -> None:
    """現行モデルの全指標を一覧表示する."""
    metrics = summarize_metrics(df)
    pe = percent_error_rates(df, thresholds=(10, 20, 30, 50))

    st.subheader("現行モデルの評価指標（最新の test 全件）")

    rows = [
        ("件数", f"{int(metrics['count']):,}", "test セットの物件数"),
        ("R² (log)", f"{metrics['r2_log']:.4f}", "log 価格スケールでの決定係数（1に近いほど良い）"),
        (
            "MAE",
            f"{format_yen_jp(metrics['mae_yen'])} ({metrics['mae_yen']:,.0f} 円)",
            "平均絶対誤差",
        ),
        (
            "RMSE",
            f"{format_yen_jp(metrics['rmse_yen'])} ({metrics['rmse_yen']:,.0f} 円)",
            "二乗平均平方根誤差。大ハズレに敏感",
        ),
        ("MAPE", f"{metrics['mape']:.3f}%", "平均絶対パーセント誤差"),
        ("Median APE", f"{metrics['median_ape']:.3f}%", "APE の中央値（外れ値に頑健）"),
        ("PE10", f"{pe['PE10']:.2f}%", "APE が 10% 以下の物件比率"),
        ("PE20", f"{pe['PE20']:.2f}%", "APE が 20% 以下の物件比率"),
        ("PE30", f"{pe['PE30']:.2f}%", "APE が 30% 以下の物件比率"),
        ("PE50", f"{pe['PE50']:.2f}%", "APE が 50% 以下の物件比率"),
        (
            "RMSE/MAE 比",
            f"{metrics['rmse_yen'] / metrics['mae_yen']:.3f}" if metrics["mae_yen"] else "—",
            "正規分布想定で 1.25。2 を超えると heavy-tail のサイン",
        ),
    ]
    summary_df = pd.DataFrame(rows, columns=["指標", "値", "意味"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _render_band_breakdown(df: pd.DataFrame) -> None:
    """価格帯別の評価指標を表示する."""
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
    for col in ("Median APE (%)", "MAPE (%)", "PE10 (%)", "PE20 (%)"):
        display[col] = display[col].round(2)

    st.subheader("価格帯別の精度")
    st.caption(
        "全体スコアは中央帯が牽引しがち。両端（低額帯・高額帯）の数値を確認することで、"
        "自動評価で使える範囲と、人手査定が望ましい範囲を見極めやすくなる。"
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_interval_summary(df: pd.DataFrame) -> None:
    """現行モデルの区間指標（PICP / PIAW）を表示する."""
    picp = coverage_rate(df)
    if math.isnan(picp):  # ※ ファイル先頭に import math を追加
        return
    widths = interval_width_stats(df)

    st.subheader("予測区間の評価（最新の test 全件）")
    rows = [
        (
            "PICP",
            f"{picp:.2f}%",
            f"実価格が予測区間に入る割合。目標(名目) {DEFAULT_NOMINAL_COVERAGE:.0f}%",
        ),
        ("PIAW(中央値)", format_yen_jp(widths["width_median_yen"]), "予測区間幅の中央値"),
        ("PIAW(平均)", format_yen_jp(widths["width_mean_yen"]), "予測区間幅の平均"),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["指標", "値", "意味"]),
        use_container_width=True,
        hide_index=True,
    )


def _render_benchmark() -> None:
    """業界 AVM のグレード基準を表示する."""
    st.subheader("業界 AVM 評価基準（NAVAR / IAAO 慣行）")
    st.caption(
        "Median APE / PE10 / PE20 をベースにしたグレード分け。"
        "Zillow Zestimate（米国）は Median APE 約 2.5% で Excellent、"
        "日本国内の AVM は MAPE 10〜15% が一般的水準。"
    )
    st.dataframe(_GRADE_BENCHMARKS, use_container_width=True, hide_index=True)


def _render_symbol_reference() -> None:
    """各指標の数式で共通的に使う記号を一覧表示する."""
    st.markdown(
        """
        各式で使う記号の意味は共通で以下の通り。

        - $n$ : 評価対象の物件数（test 件数）
        - $y_i$ : $i$ 番目の物件の**実価格**（または log 価格）
        - $\\hat{y}_i$ : $i$ 番目の物件の**予測価格**（または log 予測値）
        - $\\bar{y}$ : 実価格の平均値
        - $\\text{APE}_i = \\dfrac{|y_i - \\hat{y}_i|}{y_i} \\times 100$ : 物件 $i$ の絶対パーセント誤差
        """
    )


def _render_definitions() -> None:
    """各指標の定義・計算式・読み方の早見表を表示する."""
    st.subheader("指標の定義・計算式・読み方")
    with st.expander("📐 共通記号の意味（最初に開いて確認）", expanded=False):
        _render_symbol_reference()

    with st.expander("R² (logスケール) — モデル間の総合比較に", expanded=False):
        st.markdown("**計算式**")
        st.latex(
            r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}"
            r" \quad (\text{log スケール})"
        )
        st.markdown(
            """
            **計算手順**
            1. 各物件で残差 $y_i - \\hat{y}_i$ を計算（log 価格スケール）
            2. その二乗和（分子）と「平均からの偏差の二乗和」（分母）を取る
            3. $1 -$ 分子／分母 が決定係数

            **数値例（n=3、log 価格）**
            - 正解: $[17.5, 18.0, 19.0]$ 、予測: $[17.4, 18.1, 18.8]$
            - 残差二乗和: $0.01 + 0.01 + 0.04 = 0.06$
            - 平均からの偏差二乗和: $(17.5-18.17)^2 + (18.0-18.17)^2 + (19.0-18.17)^2 \\approx 1.17$
            - $R^2 = 1 - 0.06/1.17 \\approx 0.949$

            **意味と読み方**
            - 目的変数のばらつきのうち、モデルが説明できた割合。1.0 が完全予測。**1 に近いほど良い**。
            - **注意**: 単位を持たずモデル間比較に便利だが、これ単体では誤差の大きさ（円）が分からない。
            """
        )

    with st.expander("RMSE (logスケール) — 学習最適化の主指標"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{RMSE}_{\log} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}"
            r" \quad (\text{log スケール})"
        )
        st.markdown(
            """
            **計算手順**
            1. 各物件で log 残差 $y_i - \\hat{y}_i$ を計算
            2. 二乗和の平均（=分散の不偏でない版）
            3. 平方根を取って元のスケールに戻す

            **数値例（n=3、log 価格）**
            - 正解: $[17.5, 18.0, 19.0]$ 、予測: $[17.4, 18.1, 18.8]$
            - 残差二乗の平均: $(0.01 + 0.01 + 0.04)/3 = 0.02$
            - $\\text{RMSE}_{\\log} = \\sqrt{0.02} \\approx 0.141$

            **log → 円スケールへの直感**
            - $e^{\\text{RMSE}_{\\log}} - 1$ がおおよその比例誤差。
            - 例: $\\text{RMSE}_{\\log} = 0.30$ → $e^{0.30}-1 \\approx 0.35$ で「±35% 程度の誤差感」。

            **意味と注意**
            - log スケールでの誤差の標準的な大きさ。**小さいほど良い**。
            - 二乗するため外れ値の影響を強く受ける。
            """
        )

    with st.expander("MAE (円スケール) — 平均で何円ずれるか"):
        st.markdown("**計算式**")
        st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i| \quad (\text{円})")
        st.markdown(
            """
            **計算手順**
            1. 各物件で**絶対残差** $|y_i - \\hat{y}_i|$ を円で計算
            2. 全件平均

            **数値例（n=3、円）**
            - 正解: $[5{,}000\\text{万}, 6{,}000\\text{万}, 8{,}000\\text{万}]$
            - 予測: $[4{,}800\\text{万}, 6{,}300\\text{万}, 7{,}500\\text{万}]$
            - 絶対残差: $[200\\text{万}, 300\\text{万}, 500\\text{万}]$
            - $\\text{MAE} = (200 + 300 + 500)/3 \\approx 333\\text{万円}$

            **意味と注意**
            - 予測が平均して何円ずれているか。**小さいほど良い**。
            - 単位が「円」で最も直感的。RMSE より外れ値に頑健。
            - **高額物件と低額物件の誤差を同じ「円」で合算**するため、高額帯に引っ張られやすい。
            """
        )

    with st.expander("RMSE (円スケール) — 大ハズレの検知"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \quad (\text{円})"
        )
        st.markdown(
            """
            **計算手順**
            1. 各物件で円スケールの残差 $y_i - \\hat{y}_i$ を計算
            2. 二乗和の平均を取る
            3. 平方根

            **MAE との比較**
            - MAE は **絶対値の平均**、RMSE は **二乗の平均の平方根**。
            - 同じ誤差分布でも、RMSE は大きな外れに重みを置く。
            - 正規分布なら $\\text{RMSE}/\\text{MAE} \\approx 1.25$。**2 を超えると heavy-tail のサイン**。

            **意味と用途**
            - 円スケールでの誤差の標準的な大きさ。**小さいほど良い**。
            - 大きな外れ予測に強いペナルティを与えるため、**致命的な大ハズレの検知に向く**。
            """
        )

    with st.expander("MAPE — 相対精度の総合評価"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{y_i} \quad (\%)"
        )
        st.markdown(
            """
            **計算手順**
            1. 各物件で APE $= |y_i - \\hat{y}_i|/y_i \\times 100$ を計算（単位は %）
            2. 全件平均

            **数値例（n=3）**
            - 正解: $[5{,}000\\text{万}, 6{,}000\\text{万}, 8{,}000\\text{万}]$
            - 予測: $[4{,}800\\text{万}, 6{,}300\\text{万}, 7{,}500\\text{万}]$
            - APE: $[4.0\\%, 5.0\\%, 6.25\\%]$
            - $\\text{MAPE} = (4.0 + 5.0 + 6.25)/3 \\approx 5.08\\%$

            **意味と注意**
            - 正解額に対して**平均で何%ずれているか**。**小さいほど良い**。
            - スケールに依存しない相対指標。高額・低額物件を公平に評価できる。
            - **正解値 $y_i$ が小さい（安い）物件で値が極端に大きくなりやすく**、少数の安価物件に振られる。
            """
        )

    with st.expander("Median APE — 典型物件の体感精度"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{Median APE} = \mathrm{median}\Bigl(\{\text{APE}_i\}_{i=1}^{n}\Bigr) \quad (\%)"
        )
        st.markdown(
            r"""
            ここで $\text{APE}_i = \dfrac{|y_i - \hat{y}_i|}{y_i} \times 100$。

            **計算手順**
            1. 各物件の APE を算出
            2. 値を昇順にソート
            3. 中央の値（偶数件なら中央2件の平均）を取る

            **MAPE との違い**
            - MAPE = 平均、Median APE = 中央値。
            - $\text{MAPE} \gg \text{Median APE}$ なら大半は精度良く当てられているが一部に大ハズレ。

            **意味と長所**
            - 「典型的な（真ん中の）物件で何%ずれるか」。**小さいほど良い**。
            - 中央値なので外れ値に最も頑健。**体感精度の指標**として実用的。
            """
        )

    with st.expander("PE10 / PE20 / PE30 — 業務 KPI に直結"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{PE}X = \frac{100}{n}\sum_{i=1}^{n} \mathbb{1}\bigl[\text{APE}_i \le X\bigr]"
            r" \quad (\%)"
        )
        st.markdown(
            r"""
            ここで $\mathbb{1}[\cdot]$ は条件が真なら 1、偽なら 0 を返す**指示関数**。

            **計算手順**
            1. 各物件の APE を算出
            2. しきい値 $X$ 以下の物件を 1、それ以外を 0 とカウント
            3. 全件に対する比率を %（×100）で表示

            **数値例（n=10、しきい値 X=10）**
            - APE が 10% 以下の物件が 4 件あった場合 → $\text{PE10} = 4/10 \times 100 = 40\%$

            **業界目安**
            - PE10 ≥ 50% で Good、≥ 70% で Excellent。
            - **業務的価値**: 「自動評価でそのまま使える物件の割合」と解釈できる。
            """
        )

    with st.expander("PICP（予測区間カバレッジ率）— 区間の信頼度"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{PICP} = \frac{100}{n}\sum_{i=1}^{n}"
            r" \mathbb{1}\bigl[\,l_i \le y_i \le u_i\,\bigr] \quad (\%)"
        )
        st.markdown(
            r"""
            ここで $l_i$ / $u_i$ は物件 $i$ の予測区間の下限・上限。

            **計算手順**
            1. 各物件で実価格 $y_i$ が区間 $[l_i, u_i]$ に入るか判定（両端含む）
            2. 入った件数の割合を %（×100）で表示

            **意味と読み方**
            - 区間予測が「当たっている」割合。**名目カバレッジ（区間生成時のα、本モデルは90%）に近いほど較正が良い**。
            - 名目を**下回る**＝区間が狭すぎ（モデル過信）、**上回る**＝区間が広すぎ（情報量が低い）。
            """
        )

    with st.expander("PIAW（予測区間の幅）— 区間の鋭さ"):
        st.markdown("**計算式**")
        st.latex(
            r"\text{PIAW} = \mathrm{median}\bigl(\{\,u_i - l_i\,\}_{i=1}^{n}\bigr) \quad (\text{円})"
        )
        st.markdown(
            r"""
            **意味と読み方**
            - 予測区間の幅（上限−下限）の中央値。**狭いほど有用**だが、狭すぎると PICP が下がるトレードオフ。
            - PICP とセットで見る: 「PICP が目標近辺を保ちつつ PIAW が小さい」が理想。
            """
        )


def _render_usage_guide() -> None:
    """指標の使い分けに関するガイドを表示する."""
    st.subheader("指標の使い分け早見表")
    usage_df = pd.DataFrame(
        [
            ["R² (log)", "log", "なし", "中", "モデル間の総合比較"],
            ["RMSE (log)", "log", "log円", "低（二乗）", "学習・チューニングの最適化指標"],
            ["MAE", "円", "円", "中", "「平均で何円ずれるか」の説明"],
            ["RMSE", "円", "円", "低（二乗）", "大ハズレの検知"],
            ["MAPE", "%", "%", "低", "相対精度の総合評価"],
            ["Median APE", "%", "%", "高", "典型物件の体感精度"],
            ["PE10/PE20", "%", "%", "高", "業界 AVM 基準・運用判断"],
            ["PE10/PE20", "%", "%", "高", "業界 AVM 基準・運用判断"],
            ["PICP / PIAW", "%・円", "%・円", "—", "予測区間の較正と鋭さの評価"],
        ],
        columns=["指標", "スケール", "単位", "外れ値への頑健性", "主な用途"],
    )
    st.dataframe(usage_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        #### 読み方の指針
        1. **総合性能**は `R² (log)` と `RMSE (log)` でまず把握する。
        2. **実務的な誤差感**は `MAE`（円）と `Median APE`（%）で確認する。
        3. **大ハズレの有無**は `RMSE` と `MAE` の乖離、`MAPE` と `Median APE` の乖離で判断する。
        4. **業務 KPI**として PE10 / PE20 を併記すると、「何%の物件が自動評価可能か」が一目で分かる。
        """
    )


def main() -> None:
    """エントリポイント."""
    st.title("評価指標一覧")
    st.markdown(
        "予測モデルで利用する評価指標の定義と、現行モデルの数値、"
        "および業界 AVM 基準との位置づけをまとめたページ。"
    )
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

    _render_current_summary(df)
    st.divider()
    _render_band_breakdown(df)
    st.divider()
    _render_benchmark()
    st.divider()
    _render_definitions()
    st.divider()
    _render_usage_guide()
    _render_current_summary(df)
    st.divider()
    _render_interval_summary(df)  # 追加
    st.divider()
    _render_band_breakdown(df)


main()
