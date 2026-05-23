# 予測結果可視化BIツール 実装サマリ

不動産価格予測の結果（`outputs/test_predictions.csv`、約 7,932 件）を可視化する
**Streamlit マルチページBIツール**の実装内容をまとめる。UIに依存しない集計ロジックを
`src/visualization/` に集約し、薄いUI層（`app/pages/`）から利用する構成。

- 関連仕様: [bi_tool_spec.md](bi_tool_spec.md)（要件定義）
- 関連指標解説: [evaluation_metrics.md](evaluation_metrics.md)

---

## 1. 進め方（フェーズ）

| フェーズ | 内容 | 状態 |
|---|---|---|
| 仕様策定 | 要件定義書を作成 | ✅ 完了 |
| **P0: 基盤** | 集計ロジック（読込・行政区/駅集計） | ✅ 完了 |
| **P1: BIグラフ** | KPIサマリ＋Plotlyグラフ＋サイドバーフィルタ | ✅ 完了 |
| 追加対応 | 名称フィルタ・行政区⇄駅連動・価格帯の金額順ソート | ✅ 完了 |
| P2以降: 地図 | folium地図・クリックポップアップ | ⬜ 未着手 |

---

## 2. 成果物ファイル

| ファイル | 役割 |
|---|---|
| `docs/bi_tool_spec.md` | 要件定義書（機能/非機能/技術/実装計画/確定事項） |
| `docs/evaluation_metrics.md` | 評価指標6種の解説（KPIから参照） |
| `src/visualization/aggregate.py` | 集計・フィルタ・KPIの中核ロジック |
| `src/visualization/master.py` | 市区町村コード↔名称マスタのローダ |
| `src/visualization/format.py` | 金額の「億・万」フォーマット |
| `configs/tokyo_municipality_master.csv` | 東京都62市区町村のコード→名称マスタ |
| `app/pages/1_BIダッシュボード.py` | Streamlit BIページ本体 |
| `tests/test_aggregate.py` / `test_master.py` / `test_format.py` | ユニットテスト |

---

## 3. 公開API（純粋関数）

### `src/visualization/aggregate.py`
| 関数 | 役割 |
|---|---|
| `load_predictions` | 予測CSV読込＋必須列検証（不在は `FileNotFoundError` / 列欠落は `KeyError`） |
| `aggregate_predictions` | 汎用集計（件数・予測/実測価格の平均と中央値・MAPE・Median APE、座標指定時は `lat`/`lon` 付与） |
| `aggregate_by_ward` | 行政区（市区町村コード）単位の集計 |
| `aggregate_by_station` | 最寄駅単位の集計（代表座標つき） |
| `filter_predictions` | 多条件フィルタ（行政区/駅/種類/価格帯/面積/築年数/山手線内側） |
| `summarize_metrics` | KPI集計（R²log・MAE・RMSE・MAPE・Median APE、空データは0件＋NaN） |
| `available_stations` | 選択行政区内に存在する最寄駅一覧（連動フィルタ用） |
| `price_band_order` | 価格帯を平均実測価格の昇順（金額順）に整列 |

### `src/visualization/master.py`
| 関数 | 役割 |
|---|---|
| `load_municipality_names` | 市区町村コード→名称の対応辞書を読み込む |
| `code_to_label` | コードを表示用ラベルに変換（未知コードはコード文字列にフォールバック） |

### `src/visualization/format.py`
| 関数 | 役割 |
|---|---|
| `format_yen_jp` | 円を「億・万」併記に整形（例: `54000000` → `5,400万円`） |

---

## 4. BIページのUI機能

### サイドバーフィルタ（全グラフ・KPIに連動）
- **行政区**: 市区町村名で選択（内部ではコードに変換）
- **最寄駅**: 選択した行政区に連動して候補を絞り込み（未選択時は全駅）
- **物件種類**
- **価格帯**: 金額昇順で表示（`~2000万 → 2000万~5000万 → 5000万~1億 → 1億~3億 → 3億~`）
- **面積（㎡）/ 築年数**: レンジスライダ
- **山手線内側**: すべて / 内側のみ / 外側のみ

### KPIサマリ
件数・R²(log)・MAE・RMSE（億・万表記）・MAPE・Median APE をカード表示。

### グラフ（3タブ構成 / Plotly）
| タブ | グラフ |
|---|---|
| 予測精度 | 予測vs実測 散布図（y=x基準線・色=APE）、誤差分布ヒストグラム（APE/残差切替）、価格帯別の件数と精度 |
| エリア別 | 行政区別ランキング（区名表示）、駅別ランキング（指標切替・Top20） |
| 特徴量 | 面積 × 予測価格（色=築年数）の散布図 |

---

## 5. 主な設計判断（確認済み）

| 項目 | 決定 |
|---|---|
| アプリ構成 | マルチページ統合（`streamlit_app.py` はスタブのまま、BIは `app/pages/`） |
| 地図の初期粒度 | 行政区単位（P2で実装予定） |
| 金額表記 | 「億・万」併記 |
| 行政区フィルタ | コードではなく市区町村名で選択（利便性向上） |

### 設計上の方針
- 計算ロジックはすべて純粋関数として `src/visualization/` に集約し、UIから委譲（DRY・テスタブル）
- 価格帯の順序や名称対応は、ラベル文字列のパースに頼らず**データ／マスタ駆動**で安定化
- 散布図は4,000点上限でサンプリング（描画負荷対策）

---

## 6. 品質・検証

- ✅ `ruff format` / `ruff check` 全通過
- ✅ `pytest` → BIツール関連 **28テスト**（aggregate 20・master 6・format 2）全パス
- ✅ データ整合クロスチェック: 48市区町村コードすべてマスタ一致・住所不整合 0件
- ✅ ブラウザ実機検証（Playwright）: KPI・グラフ描画、名称フィルタ、行政区⇄駅連動、価格帯金額順をすべて確認
- ⚠️ mypy はサンドボックス環境で停滞のため未実行（型注釈・明示的例外で型安全に記述済み。`uv run mypy src/` を手元/CIで実行推奨）

---

## 7. 運用メモ

- 起動: `uv run streamlit run app/streamlit_app.py`（サイドバーから「BIダッシュボード」を選択）
- `src/` のモジュールを編集した場合は **Streamlitサーバの再起動**が必要（モジュールキャッシュのため）。ページのみ編集なら自動リロードで反映される。

---

## 8. 次のステップ候補

- **P2: 地図（駅単位）** — folium ＋ streamlit-folium 追加、駅マーカー＋クリックポップアップ
- **P3: 地図（行政区）** — 東京都GeoJSONでコロプレス＋ポップアップ（コード→名称マスタは導入済み）
- **P4** — 地図⇄グラフ連携、フィルタ後データのCSV出力 等

---

最終更新: 2026-05-24
