import random
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from src.model import OWT_LLM_v0


# 行列計算を高速化
torch.set_float32_matmul_precision("high")


# -------------------------
# デバイス取得
# -------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -------------------------
# D2Z
# -------------------------

def get_lr(it, max_lr, warmup_ratio, max_iters):

    warmup_iters = int(
        warmup_ratio * max_iters
    )

    # warmup
    # 0 → max_lr
    if it < warmup_iters:
        return max_lr * (
            it / warmup_iters
        )

    # linear decay
    # max_lr → 0
    if it < max_iters:
        progress = (
            it - warmup_iters
        ) / (
            max_iters - warmup_iters
        )

        return max_lr * (
            1.0 - progress
        )

    return 0.0


# -------------------------
# SFT用batch作成
# -------------------------

def get_batch(
    samples,
    batch_size,
    device
):

    # ランダムにbatch_size件選ぶ
    batch_samples = random.sample(
        samples,
        batch_size
    )

    # batch内で一番長いサンプル
    max_len = max(
        len(sample["input_ids"])
        for sample in batch_samples
    )

    input_ids_list = []
    labels_list = []


    for sample in batch_samples:

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # paddingする長さ
        pad_len = (
            max_len
            - len(input_ids)
        )


        # -------------------------
        # input_idsをpadding
        # -------------------------

        padded_input_ids = torch.cat([
            input_ids,

            torch.zeros(
                pad_len,
                dtype=torch.long
            )
        ])


        # -------------------------
        # labelsをpadding
        # -------------------------
        #
        # -100はloss計算から無視される
        #

        padded_labels = torch.cat([
            labels,

            torch.full(
                (pad_len,),
                -100,
                dtype=torch.long
            )
        ])


        input_ids_list.append(
            padded_input_ids
        )

        labels_list.append(
            padded_labels
        )


    # batch化
    batch_x = torch.stack(
        input_ids_list
    )

    batch_y = torch.stack(
        labels_list
    )


    return (
        batch_x.to(device),
        batch_y.to(device)
    )


# -------------------------
# 以下SFT
# -------------------------

device = get_device()


# -------------------------
# ファイル
# -------------------------

data_path = (
    "./data/sft_alpaca_train.pt"
)

# 3周Baseモデル
base_model_path = (
    "./artifacts/model_iter_32925.pt"
)

# SFT後のモデル
model_save_path = (
    "./artifacts/model_instruct.pt"
)


# -------------------------
# ハイパーパラメータ
# -------------------------

batch_size = 8

learning_rate = 1e-5

warmup_ratio = 0.02

# 52000件 ÷ 8
# = 6500 step
# 約1周相当
max_iters = 6500

grad_clip = 1.0


# -------------------------
# SFTデータ読み込み
# -------------------------

samples = torch.load(
    data_path,
    weights_only=False
)

print(
    "SFTデータ件数:",
    len(samples)
)


# -------------------------
# Baseモデル読み込み
# -------------------------

model = OWT_LLM_v0.load_from(
    base_model_path,
    device=device
)

model.train()


# -------------------------
# optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    fused=(device.type == "cuda")
)


# -------------------------
# パラメータ数確認
# -------------------------

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"パラメータ数: "
    f"{total_params:,} "
    f"({total_params / 1e6:.1f}M)"
)


# -------------------------
# SFT
# -------------------------

pbar = tqdm(
    range(max_iters)
)


for i in pbar:

    # -------------------------
    # learning rate更新
    # -------------------------

    lr = get_lr(
        i,
        learning_rate,
        warmup_ratio,
        max_iters
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


    # -------------------------
    # batch取得
    # -------------------------

    batch_x, batch_y = get_batch(
        samples,
        batch_size,
        device
    )


    # -------------------------
    # 勾配リセット
    # -------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    # -------------------------
    # 順伝播
    # -------------------------

    with autocast(
        device_type=device.type,
        dtype=torch.bfloat16
    ):

        logits = model(
            batch_x
        )


        # -------------------------
        # 1tokenずらす
        # -------------------------
        #
        # token 0を見てtoken 1を予測
        # token 1を見てtoken 2を予測
        # ...
        #

        shift_logits = logits[
            :, :-1, :
        ]

        shift_labels = batch_y[
            :, 1:
        ]


        # -------------------------
        # loss
        # -------------------------

        loss = F.cross_entropy(
            shift_logits.reshape(
                -1,
                shift_logits.size(-1)
            ),

            shift_labels.reshape(-1),

            # prompt部分とpadding部分を
            # loss計算から無視
            ignore_index=-100
        )


    # -------------------------
    # 逆伝播
    # -------------------------

    loss.backward()


    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        grad_clip
    )


    # パラメータ更新
    optimizer.step()


    # -------------------------
    # 進捗表示
    # -------------------------

    pbar.set_postfix({
        "loss":
        f"{loss.item():.4f}",

        "lr":
        f"{lr:.2e}"
    })


# -------------------------
# Instructモデル保存
# -------------------------

model.save(
    model_save_path
)


print("SFT完了")

print(
    "保存先:",
    model_save_path
)