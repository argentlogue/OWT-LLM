from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import BPETokenizer


# 学習済みBPEを読み込む
tokenizer = BPETokenizer.load_from(
    ROOT / "artifacts" / "merge_rules.pkl"
)

# owt_train.txtをtoken ID列に変換して保存
tokenizer.encode_file(
    ROOT / "data" / "owt_train.txt",
    ROOT / "data" / "owt_train.bin",
    num_processes=8
)

# owt_valid.txtをtoken ID列に変換して保存
tokenizer.encode_file(
    ROOT / "data" / "owt_valid.txt",
    ROOT / "data" / "owt_valid.bin",
    num_processes=8
)