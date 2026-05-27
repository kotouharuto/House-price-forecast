# BIツール フェーズ別フローチャート（P0〜P4）

予測結果可視化BIツールの開発フェーズと依存関係の図。

- フェーズ詳細: [bi_tool_phases.md](bi_tool_phases.md)
- 実装サマリ: [bi_tool_implementation_summary.md](bi_tool_implementation_summary.md)

> **状態の凡例**: ✅ 完了 / ⚠️ 一部未達（残課題あり） / ⬜ 未着手

```mermaid
flowchart TD
    START([BIツール開発開始]) --> P0

    subgraph P0["✅ P0: 基盤"]
        direction TB
        P0A["CSV読込<br/>BI_PREDICTIONS_PATH<br/>で切替可"]
        P0B["@st.cache_data<br/>キャッシュ化"]
        P0C["集計モジュール<br/>aggregate / master / format"]
        P0D["ユニットテスト<br/>test_aggregate<br/>test_master / test_format"]
        P0A --> P0B --> P0C --> P0D
    end

    P0 --> P1

    subgraph P1["P1: BIグラフ (機能完了)"]
        direction TB
        P1A["KPIサマリ<br/>件数・R²・MAE・RMSE<br/>MAPE・Median APE"]
        P1B["Plotlyグラフ群<br/>散布図・誤差分布・価格帯別<br/>行政区/駅ランキング・特徴量"]
        P1C["サイドバーフィルタ<br/>行政区・駅・種類・価格帯<br/>面積・築年数・山手線内側"]
        P1A & P1B & P1C
    end

    P1 --> P2

    subgraph P2["⬜ P2: 地図（駅単位）"]
        P2A["folium マーカー配置<br/>サイズ/色 = 予測価格 or APE"]
        P2B["クリックポップアップ<br/>件数・価格・APE"]
        P2C["緯度経度欠損行の除外<br/>除外件数の明示"]
        P2A --> P2B --> P2C
    end

    P2 --> P3

    subgraph P3["⬜ P3: 地図（行政区）"]
        P3A["GeoJSON調達<br/>国土数値情報"]
        P3B["市区町村コードで結合<br/>コロプレス描画"]
        P3C["粒度トグル<br/>駅 / 行政区"]
        P3D["区画クリック<br/>ポップアップ表示"]
        P3A --> P3B --> P3C --> P3D
    end

    P3 --> P4

    subgraph P4["⬜ P4: 連携・仕上げ"]
        P4A["地図 ⇄ グラフ<br/>双方向連携"]
        P4B["CSV ダウンロード<br/>フィルタ後データ"]
        P4C["ワースト物件テーブル<br/>残差上位"]
        P4D["UI 調整・全体仕上げ"]
        P4A & P4B & P4C --> P4D
    end

    P4 --> DONE([BIツール完成])

    style P0 fill:#dcfce7,stroke:#22c55e,color:#1f2937
    style P1 fill:#fef9c3,stroke:#eab308,color:#1f2937
    style P2 fill:#f3f4f6,stroke:#9ca3af,color:#1f2937
    style P3 fill:#f3f4f6,stroke:#9ca3af,color:#1f2937
    style P4 fill:#f3f4f6,stroke:#9ca3af,color:#1f2937

    classDef default fill:#ffffff,color:#1f2937,stroke:#d1d5db
```
