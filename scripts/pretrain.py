import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model import OWT_LLM_v0
torch.set_float32_matmul_precision("high") # A100でfloat32 matmulの高速化を許可

#デバイスの取得関数
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
#D2Z
def get_lr(it, max_lr, warmup_ratio, max_iters):
    # ウォームアップ：0 -> max_lr
    warmup_iters = int(warmup_ratio * max_iters)
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # アニーリング：max_lr -> 0
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0

#バッチ作成
def get_batch(data, context_len, batch_size, device, random=True, offset=0):
    if random:
        ix = torch.randint(len(data) - context_len - 1, (batch_size,))
    else:
        ix = torch.arange(offset, offset + batch_size * context_len, context_len)

        ix = ix[ix + context_len + 1 < len(data)]
        if len(ix) == 0:
            return None, None

    # バッチを作成
    x = torch.stack([torch.from_numpy(data[i:i+context_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_len+1].astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)

#evaluation
#evaluation
def evaluate(model, val_data, context_len, batch_size, device, max_batches=50):
    """Validation: 一部データを順番に処理"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    max_start = len(val_data) - context_len - 1
    num_batches = (max_start // context_len) // batch_size + 1
    num_batches = min(num_batches, max_batches)

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Validation"):
            offset = batch_idx * batch_size * context_len

            x, y = get_batch(
                val_data,
                context_len,
                batch_size,
                device,
                random=False,
                offset=offset
            )

            if x is None:
                break

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='sum'
                )

            total_loss += loss.item()
            total_tokens += y.numel()

    model.train()
    return total_loss / total_tokens

#以下、学習
# 設定
device = get_device()
data_path = './data/owt_train.bin'
val_data_path = './data/owt_valid.bin'
model_save_path = './artifacts/model_pretrain.pt'

# ハイパーパラメータ
context_len = 1024
vocab_size = 50000
batch_size = 8
learning_rate = 3e-4  # max_lr
warmup_ratio = 0.02  # ウォームアップの割合
max_iters = 54875
embed_dim = 768
n_head = 12
n_layer = 20
ff_dim = 2560
theta = 10000
eval_iters = 1500
grad_clip = 1.0
save_every = 10975 # 保存するイテレーション(約一周ごとに保存)

# データをmemmapで読み込み
train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

# モデル、オプティマイザ
model = OWT_LLM_v0(
    vocab_size, context_len, embed_dim, n_head, n_layer, ff_dim, theta
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,fused=(device.type == "cuda"))

total_params = sum(p.numel() for p in model.parameters())
print(f"パラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")

pbar = tqdm(range(max_iters))

val_loss = float('inf')
val_losses = []
val_iters = []

for i in pbar:
    # 学習率を更新
    lr = get_lr(i, learning_rate, warmup_ratio, max_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batch_x, batch_y = get_batch(train_data, context_len, batch_size, device)

    # 勾配をリセット
    optimizer.zero_grad(set_to_none=True)

    # 順伝播と損失計算(Mixed Precision)
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    # 特定のイテレーションでモデルを保存
    if (i + 1) % save_every == 0:
        save_path = f'./artifacts/model_iter_{i + 1}.pt'
        model.save(save_path)
        print(f"\nモデルを保存しました（イテレーション {i + 1}）: {save_path}")

    # 定期的に評価
    if ((i + 1) % eval_iters) == 0 or i == max_iters - 1:
        val_loss = evaluate(model, val_data, context_len, batch_size, device)
        val_losses.append(val_loss)
        val_iters.append(i + 1)
    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val_loss': f'{val_loss:.6f}'})


# Validation lossのグラフを描画
plt.figure(figsize=(10, 6))
plt.plot(val_iters, val_losses)
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.grid(True)
plt.savefig('./artifacts/loss_val.png')

model.save(model_save_path)
