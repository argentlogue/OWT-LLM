from src.model import OWT_LLM_v0
from src.tokenizer import BPETokenizer
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()

    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):

        # context長を超えたら末尾だけ使う
        input_ids = ids[:, -model.max_context_len:]

        # 現在までの文章全部をモデルに入力
        logits = model(input_ids)

        # 最後のtoken位置だけ取り出す
        logits = logits[:, -1, :]

        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 終了tokenなら終了
        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat((ids, next_id), dim=1)

    generated_ids = ids[0].tolist()

    return tokenizer.decode(generated_ids)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

#以下が生成の実行部分
# 設定
device = get_device()
model_path = "./artifacts/model_instruct.pt"
tokenizer_path = './artifacts/merge_rules.pkl'

# 生成設定
prompt1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Give three benefits of exercise.

### Response:""" # 生成の開始プロンプト

prompt2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the meaning of life?

### Response:"""
prompt3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
answer the following question: 1+1=?

### Response:"""
max_new_tokens = 80  # 生成するトークン数の上限
temperature = 0.8  # 温度パラメータ（高いほどランダム）

tokenizer = BPETokenizer.load_from(tokenizer_path)
model = OWT_LLM_v0.load_from(model_path, device=device)

# テキスト生成
print(f"--- サンプル ---")
answer = generate(
        model, tokenizer, prompt1, max_new_tokens, temperature
    )
print(answer)

print(f"--- サンプル ---")
answer = generate(
        model, tokenizer, prompt2, max_new_tokens, temperature
    )
print(answer)

print(f"--- サンプル ---")
answer = generate(
        model, tokenizer, prompt3, max_new_tokens, temperature
    )
print(answer)