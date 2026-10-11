"""PPL gate: base vs pilot, vi heldout + en sanity. Run on Vast box post-train."""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else "lfm350m-vi-cpt-pilot"

def ppl(path, texts):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype="bfloat16", device_map="auto")
    t = AutoTokenizer.from_pretrained(path)
    loss, n = 0.0, 0
    for x in texts:
        ids = t(x, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(m.device)
        with torch.no_grad():
            loss += m(ids, labels=ids).loss.item() * ids.size(1)
        n += ids.size(1)
    del m
    torch.cuda.empty_cache()
    return torch.exp(torch.tensor(loss / n)).item()

vi = open("heldout_vi.txt", encoding="utf-8").read().split("\n\n")[:200]
en = open("heldout_en.txt", encoding="utf-8").read().split("\n\n")[:200]  # prep before run
for name in ["LiquidAI/LFM2.5-350M-Base", CKPT]:
    print(f"{name}: vi={ppl(name, vi):.1f} en={ppl(name, en):.1f}")
# Go full-scale only if vi drops and en stays within ~2x.
