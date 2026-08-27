from pathlib import Path
import sys
import pickle

# プロジェクトルート
ROOT = Path(__file__).resolve().parents[1]

# srcをimportできるようにする
sys.path.insert(0, str(ROOT))

from src.tokenizer import train_bpe


# Tokenizer設定
vocab_size = 50000

# OWT
file_path = ROOT / "data" / "owt_train.txt"

# BPEを学習
merge_rules = train_bpe(
    file_path,
    vocab_size,
    num_processes=8,
    num_chunks=64
)

# 学習結果の保存場所
artifact_dir = ROOT / "artifacts"
artifact_dir.mkdir(exist_ok=True)

with open(artifact_dir / "merge_rules.pkl", "wb") as f:
    pickle.dump(merge_rules, f)