# BIツール P0/P1 改善計画

`docs/bi_tool_spec.md` を基準に、現在のBIダッシュボード実装で優先して改善したい
P0/P1項目をまとめる。Claude Codeなどに実装依頼する際の作業指示として使う。

## 目的

- P0/P1要件に対する仕様準拠度を上げる。
- 予測結果の解釈が反転するリスクをなくす。
- 入力CSVのデータ契約を明確にし、画面描画途中の `KeyError` を避ける。
- 将来的な複数CSV・複数モデル比較に備え、入力パスを設定可能にする。
- ダッシュボードの描画性能と分析しやすさを改善する。

## 対象ファイル

- `src/visualization/aggregate.py`
- `app/pages/1_BIダッシュボード.py`
- `tests/test_aggregate.py`
- 必要に応じて `docs/bi_tool_implementation_summary.md`

## ブランチ方針

1つのブランチでまとめて対応する。

```bash
git switch -c improve-bi-p0-p1-compliance
```

コミットは可能なら以下のように分ける。

1. `Fix BI prediction data contract`
2. `Improve BI dashboard caching and feature plots`
3. `Update BI tests and docs`

## 改善項目

### 1. 残差ヒストグラムの符号をCSV定義に統一する

**優先度:** P0

現在、`app/pages/1_BIダッシュボード.py` の残差ヒストグラムでは
`actual_price_yen - pred_price_yen` を使っている。一方、予測CSVの `error_yen` は
`pred_price_yen - actual_price_yen` として生成されている。

仕様書でも誤差分布ヒストグラムは `error_yen` を使う前提。

**改善内容**

- 残差ヒストグラムでは `df["error_yen"]` をそのまま使う。
- ラベルを `残差(円) = 予測 - 実測` に統一する。
- これにより、正の値は「高めに予測」、負の値は「低めに予測」と読めるようにする。

**受け入れ条件**

- APE表示は従来通り `ape_percent` を使う。
- 残差表示は `error_yen` を使う。
- UIラベルとCSV定義の符号が一致している。

### 2. `load_predictions()` の必須列チェックを拡張する

**優先度:** P0

現在の `_REQUIRED_COLUMNS` は以下の3列のみ。

- `pred_price_yen`
- `actual_price_yen`
- `ape_percent`

しかしP1画面では、フィルタ、KPI、ランキング、散布図でより多くの列を前提にしている。
このままだと古いCSVや別形式CSVを読み込んだとき、`load_predictions()` は成功し、
画面描画途中で `KeyError` が発生する。

**改善内容**

`load_predictions()` の必須列に、P0/P1で画面が前提にする列を追加する。

必須列候補:

- `actual_price_yen`
- `pred_price_yen`
- `error_yen`
- `abs_error_yen`
- `error_rate_percent`
- `ape_percent`
- `actual_price_band`
- `市区町村コード`
- `住所`
- `最寄駅：名称`
- `最寄駅：緯度`
- `最寄駅：経度`
- `種類`
- `面積（㎡）`
- `築年数`
- `都市計画`
- `最寄駅：距離（分）`
- `山手線内側`
- `actual_log_price`
- `pred_log_price`

**受け入れ条件**

- 必須列が不足しているCSVでは `load_predictions()` が `KeyError` を送出する。
- エラーメッセージに不足列名が含まれる。
- UI側で後続の `KeyError` が発生しない。
- 既存の `outputs/test_predictions.csv` は正常に読み込める。

### 3. 入力CSVパスを設定で切り替え可能にする

**優先度:** P0

仕様では、入力CSVパスは設定または環境変数で切り替え可能にする前提。
現在は `outputs/test_predictions.csv` 固定のため、別モデルや別出力ファイルの確認がしづらい。

**改善内容**

- `BI_PREDICTIONS_PATH` 環境変数をサポートする。
- 未設定時は従来通り `outputs/test_predictions.csv` を読む。
- 実装場所は `load_predictions()` または Streamlit側の `_load()` のどちらでもよいが、
  テストしやすい形にする。

**実装例**

```python
import os

def default_predictions_path() -> Path:
    return Path(os.environ.get("BI_PREDICTIONS_PATH", _DEFAULT_PRED_PATH))
```

**受け入れ条件**

- `BI_PREDICTIONS_PATH` 未設定時は既存パスを読む。
- `BI_PREDICTIONS_PATH` 設定時は指定されたCSVを読む。
- 環境変数指定時の挙動をテストで確認できる。

### 4. 集計結果キャッシュを追加する

**優先度:** P1

読み込みは `@st.cache_data` されているが、ランキング表示では
`aggregate_by_ward()` / `aggregate_by_station()` が直接呼ばれている。
仕様では読み込みだけでなく集計結果もキャッシュ対象。

**改善内容**

- Streamlit側にキャッシュ付きの集計ヘルパーを追加する。
- `_render_ranking()` から直接 `aggregate_by_ward()` / `aggregate_by_station()` を呼ばず、
  キャッシュ済みヘルパー経由にする。
- フィルタ後データに対しても正しく集計されるようにする。

**実装方針例**

```python
@st.cache_data
def _aggregate_ward_cached(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_ward(df)

@st.cache_data
def _aggregate_station_cached(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_by_station(df)
```

**受け入れ条件**

- 行政区ランキングと駅ランキングがキャッシュ済み集計を使う。
- フィルタ変更時は変更後データで再集計される。
- 表示結果が従来と同等である。

### 5. 特徴量との関係グラフを仕様に近づける

**優先度:** P1

現在は「面積 × 予測価格、色=築年数」の散布図のみ。
仕様では「面積・築年数 vs 価格」の関係を見る想定になっている。

**改善内容**

次のどちらかで対応する。

- X軸を `面積（㎡）` / `築年数` から選べる `selectbox` を追加する。
- または、`面積 × 予測価格` と `築年数 × 予測価格` の2つの散布図を表示する。

おすすめは、画面をコンパクトに保てる `selectbox` 方式。

**受け入れ条件**

- 面積と予測価格の関係を見られる。
- 築年数と予測価格の関係も見られる。
- 日本語ラベルを維持する。
- 既存のサンプリング上限 `_MAX_SCATTER_POINTS` を維持する。

## テスト方針

### 追加・更新したいテスト

- `load_predictions()` がP0/P1必須列を検証すること。
- 必須列欠落時に不足列名を含む `KeyError` を送出すること。
- `BI_PREDICTIONS_PATH` によって読み込み対象を切り替えられること。
- `price_band_order()` や既存フィルタテストが引き続き通ること。

Streamlit UI関数の細かい描画テストは必須ではないが、純粋関数化できる部分は
`src/visualization/` 側に寄せるとテストしやすい。

## 検証コマンド

```bash
uv run ruff format src/visualization app/pages tests
uv run ruff check src/visualization app/pages tests
uv run pytest tests/test_aggregate.py tests/test_master.py tests/test_format.py
```

可能なら手動でStreamlitも確認する。

```bash
uv run streamlit run app/streamlit_app.py
```

確認観点:

- BIページが起動する。
- KPIが表示される。
- 予測vs実測、誤差分布、価格帯別、行政区別、駅別、特徴量グラフが表示される。
- 残差ヒストグラムの符号説明が `予測 - 実測` になっている。
- 行政区・駅・価格帯・面積・築年数・山手線フィルタが反映される。

## 期待する完了状態

- P0/P1範囲で、仕様と実装のズレが小さくなる。
- CSVのデータ契約が明確になり、入力不備を早期検知できる。
- 残差の符号解釈が一貫する。
- 入力CSVを環境変数で切り替えられる。
- 集計処理がキャッシュされ、UIが少し軽くなる。
- 特徴量タブで面積・築年数の両方から予測価格との関係を確認できる。

