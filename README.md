# OWT-LLM

ゼロから学習した、約2.42億パラメータの英語向けDecoder-only Transformerです。

## モデル

| 種類 | 内容 |
| --- | --- |
| Base（生成コードで採用） | OpenWebTextで32,925ステップ事前学習 |
| Base（最終ステップ） | OpenWebTextで54,875ステップ事前学習 |
| Instruct | 32,925ステップ時点のBaseをStanford Alpacaで6,500ステップSFT |

重みは、このリポジトリのGitHub Releasesからダウンロードできます。

- `model_pretrain.pt`
- `model_instruct.pt`
- `model_iter_10975.pt`
- `model_iter_21950.pt`
- `model_iter_32925.pt`
- `model_iter_43900.pt`
- `model_iter_54875.pt`
- `merge_rules.pkl`

`model_iter_*`は学習過程を確認できるよう、途中チェックポイントも含めて全て公開します。`generate_base.py`とSFTの開始地点には`model_iter_32925.pt`を使用しています。`model_pretrain.pt`と`model_iter_54875.pt`のモデル重みは同一ですが、学習時に作成されたファイルとして両方を保存します。

ダウンロードしたファイルは`artifacts/`へ置いてください。

## 構成

```text
src/        モデルとtokenizer
scripts/    学習、前処理、生成
artifacts/  tokenizerと学習曲線
```

## モデル構成

- 20 layers
- hidden size 768
- 12 attention heads
- SwiGLU size 2,560
- context length 1,024
- vocabulary size 50,000
- RMSNorm / RoPE / MHA
- 241,982,208 parameters

## 実行

必要なパッケージを導入します。

```powershell
python -m pip install -r requirements.txt
```

Baseモデルの生成：

```powershell
python scripts/generate_base.py
```

Instructモデルの生成：

```powershell
python scripts/generate_instruct.py
```

## 学習データ

- 事前学習：[Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) の`train` split
- SFT：[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) の`train` split

Alpacaの52,002件中、context長を超えた2件を除く52,000件を使用しました。

## 制限

- 主に英語データで学習しています。
- context lengthは1,024です。
- 小規模な学習用モデルであり、回答の正確性は保証できません。
- InstructモデルはAlpacaのCC BY-NC 4.0条件を踏まえ、非商用の学習・研究目的を想定しています。

WebLLM版は別リポジトリ`OWT-LLM-Web`で公開します。
