"""物件査定ページのフィルタロジック（app.appraisal）のテスト.

サイドバーのスライダー操作＝範囲条件の適用なので、``apply_range`` /
``apply_filters`` を市区町村ごとに検証することで、フィルタが正しく
動作しているかを確認する。
"""

import numpy as np
import pandas as pd
import pytest

from app.appraisal import (
    AGE_COL,
    AREA_COL,
    PRICE_COL,
    STATION_DISTANCE_COL,
    TYPE_COL,
    WARD_CODE_COL,
    apply_filters,
    apply_range,
    build_candidate_pool,
    property_label,
)

# テスト対象の市区町村コード（千代田区・渋谷区・世田谷区を模した3区）
_WARDS = (13101, 13113, 13112)


def _make_multi_ward_df(rows_per_ward: int = 20, seed: int = 0) -> pd.DataFrame:
    """複数市区町村・複数種類を含む取引データを生成する.

    市区町村ごとに面積・築年数・駅距離・価格の水準をずらし、フィルタが
    区をまたいで混線しないことを検証できるようにする。
    """
    rng = np.random.default_rng(seed)
    frames = []
    for offset, ward in enumerate(_WARDS):
        n = rows_per_ward
        frames.append(
            pd.DataFrame(
                {
                    WARD_CODE_COL: ward,
                    TYPE_COL: "中古マンション等",
                    # 区ごとに面積帯をずらす（offset*10 ㎡ 加算）
                    AREA_COL: rng.uniform(30, 90, n) + offset * 10,
                    AGE_COL: rng.integers(1, 50, n).astype(float),
                    STATION_DISTANCE_COL: rng.integers(1, 20, n).astype(float),
                    # 区ごとに価格帯をずらす（offset 億円加算）
                    PRICE_COL: rng.uniform(2_000, 8_000, n) * 1e4 + offset * 1e8,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    # 別種類の行も混ぜる（種類フィルタの確認用）
    other = df.head(5).copy()
    other[TYPE_COL] = "宅地(土地)"
    return pd.concat([df, other], ignore_index=True)


def test_build_candidate_pool_filters_ward_and_type() -> None:
    """行政区と種類の両方で母集団が絞られること."""
    df = _make_multi_ward_df()

    pool = build_candidate_pool(df, ward_code=13113, property_type="中古マンション等")

    assert (pool[WARD_CODE_COL] == 13113).all()
    assert (pool[TYPE_COL] == "中古マンション等").all()
    assert not pool.empty


@pytest.mark.parametrize("ward", _WARDS)
def test_apply_range_area_keeps_only_in_range_per_ward(ward: int) -> None:
    """市区町村ごとに面積スライダーを動かすと範囲内の物件だけが残ること."""
    df = _make_multi_ward_df()
    pool = build_candidate_pool(df, ward, "中古マンション等")

    # その区の面積分布の中央付近に範囲を絞る
    lo = float(pool[AREA_COL].quantile(0.25))
    hi = float(pool[AREA_COL].quantile(0.75))
    result = apply_range(pool, AREA_COL, lo, hi)

    # 残った行はすべて範囲内、かつ母集団より件数が減る
    assert result[AREA_COL].between(lo, hi).all()
    assert len(result) < len(pool)
    # 範囲外の行が確かに除外されている
    excluded = pool[~pool.index.isin(result.index)]
    assert (~excluded[AREA_COL].between(lo, hi)).all()


@pytest.mark.parametrize("ward", _WARDS)
@pytest.mark.parametrize("column", [AREA_COL, AGE_COL, STATION_DISTANCE_COL, PRICE_COL])
def test_apply_range_each_item_per_ward(ward: int, column: str) -> None:
    """各項目（面積・築年数・駅徒歩・価格）のスライダーが区ごとに機能すること."""
    df = _make_multi_ward_df()
    pool = build_candidate_pool(df, ward, "中古マンション等")

    lo = float(pool[column].quantile(0.30))
    hi = float(pool[column].quantile(0.70))
    result = apply_range(pool, column, lo, hi)

    assert result[column].between(lo, hi).all()
    # 他の区の物件が混入していないこと
    assert (result[WARD_CODE_COL] == ward).all()


def test_apply_range_widening_returns_more_or_equal() -> None:
    """範囲を広げるほど該当件数が単調に増える（減らない）こと."""
    df = _make_multi_ward_df()
    pool = build_candidate_pool(df, 13101, "中古マンション等")

    narrow = apply_range(pool, AGE_COL, 20, 25)
    wide = apply_range(pool, AGE_COL, 10, 40)

    assert len(narrow) <= len(wide)
    assert set(narrow.index).issubset(set(wide.index))


def test_apply_range_full_range_keeps_all() -> None:
    """min〜max のフル範囲なら全件残ること（スライダー初期状態）."""
    df = _make_multi_ward_df()
    pool = build_candidate_pool(df, 13112, "中古マンション等")

    lo, hi = float(pool[PRICE_COL].min()), float(pool[PRICE_COL].max())
    result = apply_range(pool, PRICE_COL, lo, hi)

    assert len(result) == len(pool)


def test_apply_range_excludes_nan_rows() -> None:
    """範囲条件は欠損値を持つ行を除外すること."""
    pool = pd.DataFrame({AREA_COL: [40.0, np.nan, 60.0]})

    result = apply_range(pool, AREA_COL, 0.0, 100.0)

    assert len(result) == 2
    assert result[AREA_COL].notna().all()


def test_apply_filters_combines_all_conditions() -> None:
    """複数項目を同時に適用するとすべての条件を満たす行だけ残ること."""
    df = _make_multi_ward_df()
    pool = build_candidate_pool(df, 13113, "中古マンション等")

    ranges = {
        AREA_COL: (float(pool[AREA_COL].quantile(0.2)), float(pool[AREA_COL].quantile(0.9))),
        AGE_COL: (5.0, 40.0),
        PRICE_COL: (float(pool[PRICE_COL].quantile(0.1)), float(pool[PRICE_COL].quantile(0.8))),
    }
    result = apply_filters(pool, ranges)

    for column, (lo, hi) in ranges.items():
        assert result[column].between(lo, hi).all()
    # 個別適用の積集合と一致すること
    manual = pool
    for column, (lo, hi) in ranges.items():
        manual = apply_range(manual, column, lo, hi)
    assert set(result.index) == set(manual.index)


def test_apply_filters_empty_when_no_match() -> None:
    """どの行も満たさない条件なら空になること."""
    df = _make_multi_ward_df()
    pool = build_candidate_pool(df, 13101, "中古マンション等")

    # 存在しない価格帯（負の範囲）を指定
    result = apply_filters(pool, {PRICE_COL: (-100.0, -1.0)})

    assert result.empty


def test_property_label_contains_key_attributes() -> None:
    """ラベルに面積・築年数・駅距離・価格が含まれること."""
    row = pd.Series(
        {
            AREA_COL: 60.0,
            AGE_COL: 7.0,
            STATION_DISTANCE_COL: 5.0,
            PRICE_COL: 32_000_000,
        }
    )

    label = property_label(row)

    assert "60㎡" in label
    assert "築7年" in label
    assert "駅5分" in label
    assert "万円" in label
