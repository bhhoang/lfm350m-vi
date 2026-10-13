"""Prep Vietnamese SFT data from aya_dataset (human-annotated multilingual instruct)."""
import json
import random
from datasets import load_dataset

random.seed(0)
ds = load_dataset("CohereForAI/aya_dataset", split="train")
vi = ds.filter(lambda x: x["language"] == "vie")
print("vi rows:", len(vi))
vi = vi.filter(lambda x: len(x["inputs"]) < 2000 and len(x["targets"]) < 3000)
print("after len filter:", len(vi))


def fmt(x):
    return {"text": "### Instruction:\n" + x["inputs"].strip()
            + "\n\n### Response:\n" + x["targets"].strip()}


vi = vi.map(fmt, remove_columns=vi.column_names).shuffle(seed=0)
n_val = max(100, int(len(vi) * 0.03))
with open("sft_val.jsonl", "w", encoding="utf-8") as f:
    for r in vi.select(range(n_val)):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("sft_vi.jsonl", "w", encoding="utf-8") as f:
    for r in vi.select(range(n_val, len(vi))):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train:", len(vi) - n_val, "val:", n_val)
