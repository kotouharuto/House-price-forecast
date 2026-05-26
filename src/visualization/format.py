"""表示用のフォーマット関数を提供するモジュール（UI非依存）."""

import math


def format_yen_jp(value: float) -> str:
    """円の数値を「億・万」併記の文字列に変換する.

    万単位に丸めて表示する（例: ``54000000`` -> ``"5,400万円"``、
    ``120000000`` -> ``"1億2,000万円"``）。``NaN`` / ``None`` は ``"—"`` を返す。
    負値は先頭に ``"-"`` を付ける。

    Args:
        value: 円単位の金額。

    Returns:
        「億・万」併記の表示用文字列。
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"

    sign = "-" if value < 0 else ""
    # 万単位に丸める（万未満は四捨五入）
    man_total = round(abs(value) / 1e4)
    oku, man = divmod(man_total, 10000)

    if oku and man:
        body = f"{oku:,}億{man:,}万円"
    elif oku:
        body = f"{oku:,}億円"
    else:
        body = f"{man:,}万円"

    return sign + body
