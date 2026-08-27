import pandas as pd
import torch

from src.tokenizer import BPETokenizer


# -------------------------
# 設定
# -------------------------

data_path = "./data/sft_alpaca_train.parquet"
tokenizer_path = "./artifacts/merge_rules.pkl"
output_path = "./data/sft_alpaca_train.pt"

max_context_len = 1024


# -------------------------
# データとtokenizerを読み込む
# -------------------------

df = pd.read_parquet(data_path)

tokenizer = BPETokenizer.load_from(tokenizer_path)


# SFT用データを入れるリスト
samples = []

# 長すぎて除外した件数
skipped = 0


# -------------------------
# 1件ずつSFT用データに変換
# -------------------------

for _, row in df.iterrows():

    instruction = row["instruction"]
    input_text = row["input"]
    response = row["output"]


    # inputがある場合
    if input_text.strip():

        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            + instruction
            + "\n\n"
            "### Input:\n"
            + input_text
            + "\n\n"
            "### Response:\n"
        )

    # inputがない場合
    else:

        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            + instruction
            + "\n\n"
            "### Response:\n"
        )


    # promptとresponseを別々にtokenize
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)


    # responseの最後に終了tokenを追加
    response_ids.append(tokenizer.end_token_id)


    # モデルに入力するtoken列
    input_ids = prompt_ids + response_ids


    # 1024 tokenを超えるデータは今回は使わない
    if len(input_ids) > max_context_len:
        skipped += 1
        continue


    # -------------------------
    # labelsを作る
    # -------------------------
    #
    # prompt部分:
    #   モデルには読ませる
    #   しかしlossは計算しない
    #
    # response部分:
    #   lossを計算する
    #
    # -100はCrossEntropyLossで無視させるための値
    #

    labels = (
        [-100] * len(prompt_ids)
        + response_ids
    )


    # tensorにして保存
    sample = {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long
        ),

        "labels": torch.tensor(
            labels,
            dtype=torch.long
        )
    }

    samples.append(sample)


# -------------------------
# 保存
# -------------------------

torch.save(samples, output_path)


# -------------------------
# 結果確認
# -------------------------

print("SFTデータ作成完了")
print("元データ件数:", len(df))
print("使用データ件数:", len(samples))
print("除外件数:", skipped)
print("保存先:", output_path)