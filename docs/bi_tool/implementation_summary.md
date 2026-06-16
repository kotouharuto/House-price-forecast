# 予測結果可視化BIツール 実装サマリ（最終版）

東京都不動産価格予測モデル（`outputs/test_predictions.csv`、7,932 件）の予測結果を
可視化する Streamlit マルチページBIツールについて、計画フェーズ P0 から P4 までの
全実装が完了した時点での内容を整理する。

UIに依存しない集計・ローディングロジックを `src/visualization/` に集約し、
Streamlit のページ層（`app/pages/`）はその上に立つ薄いUIとして実装した。
ページ間で共通利用するフィルタは `app/filters.py` に切り出している。

- 関連仕様: [spec.md](spec.md)
- フェーズ別スコープ: [phases.md](phases.md)
- P0/P1 改善計画: [p0_p1_improvement_plan.md](p0_p1_improvement_plan.md)
- 指標定義: [../modeling/evaluation_metrics.md](../modeling/evaluation_metrics.md)

---

## 1. フェーズ別の進捗

| フェーズ | 内容 | 対応要件 | 状態 |
|---|---|---|---|
| 仕様策定 | 要件定義書の作成 | - | 完了 |
| P0: 基盤 | データ読込・キャッシュ・集計モジュール・ユニットテスト | M-1 | 完了 |
| P0 改善 | 残差符号の統一・必須列拡張・入力CSV切替 | M-1 | 完了 |
| P1: BIグラフ | KPIサマリ・Plotlyグラフ群・サイドバーフィルタ | M-2, M-3, M-6 | 完了 |
| P1 改善 | 集計キャッシュ・特徴量X軸切替・UX調整 | M-2, M-3, M-6, §3.3 | 完了 |
| P2: 地図（駅単位） | folium マーカー＋クリックポップアップ | M-4(駅), M-5(駅) | 完了 |
| P3: 地図（行政区） | GeoJSON 同梱＋コロプレス＋ポップアップ＋粒度トグル | M-4(区), M-5(区) | 完了 |
| P4: 連携・仕上げ | 地図クリック連動・CSV出力・ワースト物件 | N-1, N-2, N-3 | 完了 |

---

## 2. 成果物一覧

### ドキュメント
| ファイル | 役割 |
|---|---|
| `docs/spec.md` | 要件定義書（機能・非機能・実装計画） |
| `docs/phases.md` | フェーズ別スコープと現状の状態 |
| `docs/p0_p1_improvement_plan.md` | P0/P1 の改善作業指示書 |
| `docs/implementation_summary.md` | 本文書（最終サマリ） |
| `docs/../modeling/evaluation_metrics.md` | 評価指標の定義 |

### 集計・ロジック層（UI非依存）
| ファイル | 役割 |
|---|---|
| `src/visualization/aggregate.py` | 集計・フィルタ・KPI・ワースト抽出の中核 |
| `src/visualization/geo.py` | GeoJSON ローダ（行政区ポリゴン／駅 Point／物件 Point） |
| `src/visualization/master.py` | 市区町村コード↔名称マスタの読み込み |
| `src/visualization/format.py` | 金額の「億・万」表示整形 |
| `configs/tokyo_municipality_master.csv` | 東京都市区町村のコード↔名称マスタ |
| `configs/tokyo_municipalities.geojson` | 行政区ポリゴンの GeoJSON |

### UI 層
| ファイル | 役割 |
|---|---|
| `app/filters.py` | サイドバーフィルタ（BIページ・地図ページ共通） |
| `app/pages/1_BIダッシュボード.py` | KPI・グラフ・ワースト物件・CSV出力 |
| `app/pages/2_地図.py` | 行政区／駅の地図ページ（粒度トグル） |
| `app/__init__.py` | アプリ層をパッケージ化（共通モジュール import 用） |

### 補助スクリプト
| ファイル | 役割 |
|---|---|
| `scripts/csv_to_geojson.py` | 予測結果CSVを物件単位／駅単位の Point GeoJSON に変換 |

### テスト
| ファイル | 役割 |
|---|---|
| `tests/test_aggregate.py` | 集計・フィルタ・KPI・ワースト抽出のテスト |
| `tests/test_master.py` | 市区町村マスタのテスト |
| `tests/test_format.py` | 金額整形のテスト |
| `tests/test_geo.py` | GeoJSON ローダ3種のテスト |
| `tests/test_csv_to_geojson.py` | CSV→GeoJSON 変換のテスト |

---

## 3. 公開 API（純粋関数）

### `src/visualization/aggregate.py`
| 関数 | 役割 |
|---|---|
| `load_predictions` | 予測CSV読込＋必須列検証。`BI_PREDICTIONS_PATH` 環境変数で入力切替可 |
| `default_predictions_path` | 既定の入力CSVパスを返す（環境変数優先） |
| `aggregate_predictions` | 汎用集計（件数・平均／中央予測実測・MAPE・Median APE、座標指定時は lat/lon 付与） |
| `aggregate_by_ward` | 行政区単位の集計 |
| `aggregate_by_station` | 駅単位の集計（代表座標つき） |
| `station_map_summary` | 駅単位集計＋代表種類／価格帯（地図ポップアップ用） |
| `ward_map_summary` | 行政区単位集計＋代表種類／価格帯（コロプレスポップアップ用） |
| `filter_predictions` | 多条件フィルタ（行政区／駅／種類／価格帯／面積／築年数／山手線内側） |
| `summarize_metrics` | KPI集計（R²log・MAE・RMSE・MAPE・Median APE） |
| `available_stations` | 選択行政区内の駅一覧（連動フィルタ用） |
| `price_band_order` | 価格帯ラベルを平均実測価格の昇順に整列 |
| `worst_properties` | 残差絶対値またはAPE降順で上位N件を抽出 |

### `src/visualization/geo.py`
| 関数 | 役割 |
|---|---|
| `load_municipality_geojson` | 行政区ポリゴンGeoJSONを読み込み、市区町村コードプロパティを検証 |
| `load_station_geojson` | `scripts/csv_to_geojson.py` 生成の駅Point GeoJSONを読み込み・検証 |
| `load_property_geojson` | 同上、物件Point GeoJSONを読み込み・検証 |

### `src/visualization/master.py`
| 関数 | 役割 |
|---|---|
| `load_municipality_names` | 市区町村コード→名称の対応辞書を返す |
| `code_to_label` | コードを表示用ラベル（市区町村名）に変換 |

### `src/visualization/format.py`
| 関数 | 役割 |
|---|---|
| `format_yen_jp` | 円の数値を「億・万」併記の文字列に変換 |

---

## 4. UI 機能

### サイドバーフィルタ（共通）
全フィルタにウィジェットキー（`flt_*`）を付与しており、BIページと地図ページの
間で選択状態が共有される。

- 行政区（市区町村名で選択、内部はコードで保持）
- 最寄駅（選択中の行政区に連動して候補を絞り込み）
- 物件種類
- 価格帯（金額昇順）
- 面積（㎡）、築年数（レンジスライダ）
- 山手線内側（すべて／内側のみ／外側のみ）

### BIページ（`1_BIダッシュボード.py`）
- KPIサマリ: 件数、R²(log)、MAE、RMSE（億・万表示）、MAPE、Median APE。
  各カードに指標定義のツールチップを付与。
- ダウンロード（N-2）: フィルタ後生データ、行政区集計、駅集計を CSV で出力
  （UTF-8 BOM 付きで Excel 互換）。
- タブ構成:
  - 予測精度: 散布図（y=x基準線、色＝APE、ホバーに億・万表示）、
    誤差分布ヒストグラム（APE／残差(`error_yen`)切替）、価格帯別件数と精度
  - エリア別: 行政区ランキング、駅ランキング（指標切替、Top20、棒に億・万または％ラベル、
    色＝平均APE）
  - 特徴量: 面積／築年数 × 予測価格の散布図（X軸切替、色＝もう一方）
  - ワースト物件（N-3）: 残差絶対値またはAPEで上位N件をテーブル表示
    （ソート基準・件数を切替可、行政区はコード→名称に変換）

### 地図ページ（`2_地図.py`）
- 粒度トグル: 行政区／駅（初期表示は行政区、仕様準拠）
- 色分け指標: 平均予測価格／平均実測価格／平均APE
- 行政区ブランチ: GeoJSONによるコロプレス＋透明レイヤのクリックポップアップ
  （行政区名・件数・予測／実測の平均と中央値・APE・代表種類／価格帯）。
  GeoJSON未配置時は親切なエラー表示と切替誘導でクラッシュを回避。
- 駅ブランチ: CircleMarker（色＝指標、半径＝件数の平方根スケール）＋クリックポップアップ。
  緯度経度欠損駅は除外し件数を明示。
- 地図クリック連動（N-1）: 区画／マーカーをクリックすると、対応する行政区／駅フィルタを
  「置換」セマンティクスで更新（行政区クリック時は駅フィルタもクリア）。
  サイドバーは即時反映され、BIページのグラフにも同じフィルタが効く。
  「クリック選択をクリア」ボタンで一括解除可能。

---

## 5. 主な設計判断

| 項目 | 決定 |
|---|---|
| アプリ構成 | Streamlit マルチページ。エントリは `app/home.py`、コンテンツは `app/pages/` |
| ロジックとUIの分離 | 集計・ローディングは `src/visualization/`、UIは `app/` |
| ページ間の共通化 | フィルタは `app/filters.py` の `render_sidebar_filters` に一本化（DRY） |
| ページ間の状態共有 | フィルタウィジェットに明示キー（`flt_*`）を付与し、Streamlit の session_state で共有 |
| 入力CSV切替 | 環境変数 `BI_PREDICTIONS_PATH` で差し替え可能 |
| 集計のキャッシュ | 読み込み・集計・GeoJSON は `@st.cache_data` を適用 |
| 必須列の早期検証 | `load_predictions` でダッシュボードが参照する13列を読込時に検証し、後段の `KeyError` を防止 |
| 誤差符号の統一 | 残差は CSV の `error_yen`（= 予測 − 実測）に統一 |
| 金額表記 | KPI とランキング棒・ホバーで「億・万」併記、散布図軸は円（ホバーで補助） |
| 地図クリックの意味 | 仕様の「双方向連携」は「マップクリックでフィルタを置換」「フィルタ変更で地図も再描画」として実現 |
| 地図ライブラリ | folium ＋ streamlit-folium。ポップアップは folium 標準機能で実現し、戻り値取得はクリック連動のみに使用 |
| GeoJSON管理 | 行政区ポリゴンは `configs/` に配置（コミット済前提）、駅／物件 Point は `scripts/csv_to_geojson.py` で生成可能 |

---

## 6. 品質・検証

### 自動チェック
- `uv run ruff format` / `uv run ruff check` 全通過
- `uv run pytest` 63件全パス（内訳）
  - `test_aggregate.py`: 集計・フィルタ・KPI・ワースト抽出など30件
  - `test_geo.py`: GeoJSON ローダ3種10件
  - `test_master.py`: 市区町村マスタ6件
  - `test_format.py`: 金額整形2件
  - `test_csv_to_geojson.py`: CSV→GeoJSON 変換3件
  - その他（前処理・設定など）: 12件

### スモークテスト（実データ）
- 駅集計: 559駅中538駅をプロット、緯度経度欠損21駅を地図から除外
- 行政区集計: 48行政区を集計、`folium.Choropleth` で塗り分けまで成功
- ワースト物件: 残差絶対値で上位N件抽出を確認（半蔵門・広尾の高額帯が上位）
- CSV→GeoJSON 変換: 物件 Point 7,772件・駅 Point 538件を生成

### 補足
- mypy は環境（pandas-stubs 未導入）に起因するエラーが残るのみで、コード自体は型注釈済み
- ブラウザでの最終的なUI挙動確認は、各PRマージ前に手動で実施することを推奨

---

## 7. 運用メモ

### アプリ起動
```
uv run streamlit run app/home.py
```
サイドバーから「BIダッシュボード」「地図」のページを切替。

### 別モデルの予測結果を表示
```
BI_PREDICTIONS_PATH=/path/to/other_predictions.csv \
  uv run streamlit run app/home.py
```

### CSV→GeoJSON 変換
```
uv run python scripts/csv_to_geojson.py
```
（既定で `outputs/test_predictions.csv` をコピーし、物件／駅 Point GeoJSON を `outputs/` に出力）

### 行政区 GeoJSON の配置
`configs/tokyo_municipalities.geojson` に東京都市区町村の GeoJSON（国土数値情報 N03 等）
を配置する必要がある。`features[].properties.N03_007` に JIS5桁の市区町村コードを
文字列で持つことが前提。プロパティ名が異なる場合は
`load_municipality_geojson(..., code_property="...")` で上書き可能。

### `src/` の編集時の注意
`@st.cache_data` 等のモジュールキャッシュの影響で、`src/visualization/` を編集した場合は
Streamlit サーバの再起動が必要。`app/pages/` 内の編集はホットリロードで反映される。

---

## 8. 既知の制約と今後の検討事項

### 性能
- `configs/tokyo_municipalities.geojson` の頂点数が多い場合、コロプレスの初回描画が
  遅くなる。`mapshaper` での `-dissolve2 N03_007` と `-simplify` による軽量化を推奨。

### 機能面
- 任意物件のインタラクティブ予測（F-1）、時系列トレンド（F-2）、複数モデル比較（F-3）、
  Streamlit Cloud デプロイ（F-4）は仕様の Future Enhancement として未着手。

### データ
- `configs/tokyo_municipalities.geojson` および `scripts/csv_to_geojson.py` の生成物
  （`outputs/test_predictions_copy.csv`、`outputs/test_predictions_*.geojson`）は
  リポジトリ運用上、サイズや `.gitignore` の扱いを必要に応じて見直す余地がある。

---

最終更新: 2026-05-31
