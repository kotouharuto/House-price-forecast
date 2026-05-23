"""表示フォーマット（src.visualization.format）のテスト."""

import pytest

from src.visualization.format import format_yen_jp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (54_000_000, "5,400万円"),
        (120_000_000, "1億2,000万円"),
        (100_000_000, "1億円"),
        (57_056_387, "5,706万円"),  # 万未満は四捨五入
        (0, "0万円"),
        (-5_000_000, "-500万円"),
    ],
)
def test_format_yen_jp_formats_values(value: float, expected: str) -> None:
    """円の数値が「億・万」併記の文字列に変換されること."""
    assert format_yen_jp(value) == expected


def test_format_yen_jp_handles_nan() -> None:
    """NaN は '—' を返すこと."""
    assert format_yen_jp(float("nan")) == "—"
