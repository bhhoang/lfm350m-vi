"""Extended eval: PPL (base vs LoRA) + generation samples. Run on Vast box post-train."""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "LiquidAI/LFM2.5-350M-Base"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "lfm350m-vi-cpt-pilot"


def load(path=None):
    m = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto")
    if path:
        m = PeftModel.from_pretrained(m, path)
        m = m.merge_and_unload()
    t = AutoTokenizer.from_pretrained(BASE)
    m.eval()
    return m, t


def ppl(model, tok, texts):
    loss, n = 0.0, 0
    for x in texts:
        ids = tok(x, return_tensors="pt", truncation=True,
                  max_length=2048).input_ids.to(model.device)
        with torch.no_grad():
            loss += model(ids, labels=ids).loss.item() * ids.size(1)
        n += ids.size(1)
    return torch.exp(torch.tensor(loss / n)).item()


vi = open("heldout_vi.txt", encoding="utf-8").read().split("\n\n")[:200]
en = open("heldout_en.txt", encoding="utf-8").read().split("\n\n")[:200]

m0, t0 = load()
print(f"base: vi={ppl(m0, t0, vi):.1f} en={ppl(m0, t0, en):.1f}", flush=True)
del m0
torch.cuda.empty_cache()

m1, t1 = load(CKPT)
print(f"tuned: vi={ppl(m1, t1, vi):.1f} en={ppl(m1, t1, en):.1f}", flush=True)

# Generation samples: heldout prefixes, greedy, short — eyeball Vietnamese fluency
print("\n--- samples (tuned) ---", flush=True)
for x in vi[:10]:
    prefix = x[:120]
    ids = t1(prefix, return_tensors="pt").input_ids.to(m1.device)
    with torch.no_grad():
        out = m1.generate(ids, max_new_tokens=60, do_sample=False)
    print("PREFIX:", prefix.replace("\n", " ")[:100], flush=True)
    print("GEN:", t1.decode(out[0][ids.size(1):],
                            skip_special_tokens=True).replace("\n", " ")[:200], flush=True)
    print(flush=True)
