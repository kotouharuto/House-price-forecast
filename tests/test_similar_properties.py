"""類似物件検索（src.visualization.similar_properties）のテスト."""

import numpy as np
import pandas as pd
import pytest

from src.visualization.similar_properties import find_similar_properties


def _make_sample_df() -> pd.DataFrame:
    """テスト用の小さな取引データを生成する.

    対象物件 idx=0 と同じ種類・市区町村のものを複数件、別の市区町村のものを
    数件混在させ、距離・取引時期の振る舞いを検証可能な形にする。
    """
    return pd.DataFrame(
        {
            "種類": [
                "中古マンション等",
                "中古マンション等",
                "中古マンション等",
                "中古マンション等",
                "中古マンション等",
                "中古マンション等",  # 別市区町村
                "宅地（土地）",  # 種類違い
            ],
            "市区町村コード": [13112, 13112, 13112, 13112, 13112, 13109, 13112],
            "面積（㎡）": [60.0, 62.0, 58.0, 80.0, 30.0, 60.0, 60.0],
            "築年数": [20.0, 18.0, 22.0, 25.0, 5.0, 20.0, 20.0],
            "最寄駅：距離（分）": [5.0, 4.0, 6.0, 8.0, 3.0, 5.0, 5.0],
            "取引年": [2024, 2024, 2024, 2023, 2024, 2024, 2024],
            "取引四半期": [3, 4, 1, 2, 2, 3, 3],
            "actual_price_yen": [
                52_000_000,
                58_000_000,
                52_000_000,
                70_000_000,
                36_000_000,
                52_000_000,
                52_000_000,
            ],
        }
    )


def test_returns_n_neighbors_from_same_category() -> None:
    """同じ種類・市区町村から指定件数が返ること."""
    df = _make_sample_df()

    result = find_similar_properties(df, target_idx=0, n_neighbors=3)

    assert len(result) == 3
    assert (result["種類"] == "中古マンション等").all()
    assert (result["市区町村コード"] == 13112).all()


def test_excludes_target_itself() -> None:
    """対象物件自身が結果に含まれないこと."""
    df = _make_sample_df()

    result = find_similar_properties(df, target_idx=0, n_neighbors=5)

    assert 0 not in result.index


def test_filter_excludes_different_category() -> None:
    """フィルタ列の値が異なる物件は結果に含まれないこと."""
    df = _make_sample_df()

    # n_neighbors を多めに指定しても、母集団は同一フィルタ内に限定される
    result = find_similar_properties(df, target_idx=0, n_neighbors=10)

    # 別市区町村 (idx=5) と種類違い (idx=6) は除外されている
    assert 5 not in result.index
    assert 6 not in result.index


def test_nearest_first_when_recent_first_false() -> None:
    """recent_first=False のとき距離の近い順で並ぶこと."""
    df = _make_sample_df()

    result = find_similar_properties(df, target_idx=0, n_neighbors=4, recent_first=False)

    # 距離が単調非減少（近い順）になっている
    distances = result["similarity_distance"].to_numpy()
    assert (np.diff(distances) >= 0).all()


def test_recent_first_orders_by_period_desc() -> None:
    """recent_first=True のとき取引年・四半期の降順で並ぶこと."""
    df = _make_sample_df()

    result = find_similar_properties(df, target_idx=0, n_neighbors=4, recent_first=True)

    # (取引年, 取引四半期) のタプルが降順
    pairs = list(zip(result["取引年"], result["取引四半期"], strict=True))
    assert pairs == sorted(pairs, reverse=True)


def test_raises_when_target_idx_missing() -> None:
    """存在しない target_idx で ValueError."""
    df = _make_sample_df()

    with pytest.raises(ValueError, match="target_idx"):
        find_similar_properties(df, target_idx=9999, n_neighbors=3)


def test_raises_when_pool_empty() -> None:
    """フィルタ後の母集団が空のとき ValueError."""
    # 対象物件が「唯一の組合せ」になるようなデータ
    df = pd.DataFrame(
        {
            "種類": ["中古マンション等", "中古マンション等"],
            "市区町村コード": [13112, 13109],  # 全部別市区町村
            "面積（㎡）": [60.0, 70.0],
            "築年数": [20.0, 25.0],
            "最寄駅：距離（分）": [5.0, 6.0],
            "取引年": [2024, 2024],
            "取引四半期": [3, 1],
        }
    )

    with pytest.raises(ValueError, match="母集団が空"):
        find_similar_properties(df, target_idx=0, n_neighbors=3)


def test_raises_when_required_column_missing() -> None:
    """必須列が無いとき KeyError."""
    df = pd.DataFrame(
        {
            "種類": ["中古マンション等", "中古マンション等"],
            # 市区町村コード が無い
            "面積（㎡）": [60.0, 70.0],
            "築年数": [20.0, 25.0],
            "最寄駅：距離（分）": [5.0, 6.0],
            "取引年": [2024, 2024],
            "取引四半期": [3, 1],
        }
    )

    with pytest.raises(KeyError, match="市区町村コード"):
        find_similar_properties(df, target_idx=0, n_neighbors=3)


def test_drops_rows_with_missing_distance_features() -> None:
    """距離計算列に欠損がある行は母集団から除外されること."""
    df = _make_sample_df()
    # 同一フィルタ内の 1 件 (idx=1) の面積を欠損にする
    df.loc[1, "面積（㎡）"] = np.nan

    result = find_similar_properties(df, target_idx=0, n_neighbors=10)

    assert 1 not in result.index


def test_returns_all_when_n_neighbors_exceeds_pool() -> None:
    """要求件数が母集団より多い場合は全件返すこと."""
    df = _make_sample_df()
    # 同一フィルタ内は idx 0, 1, 2, 3, 4 の 5 件（うち対象除外で 4 件）
    result = find_similar_properties(df, target_idx=0, n_neighbors=100)

    assert len(result) == 4


def test_similarity_distance_column_added() -> None:
    """結果 DataFrame に similarity_distance 列が追加されること."""
    df = _make_sample_df()

    result = find_similar_properties(df, target_idx=0, n_neighbors=3)

    assert "similarity_distance" in result.columns
    assert (result["similarity_distance"] >= 0).all()
