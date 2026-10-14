"""Pull bigger mixed VN SFT data: aya + Vistral + translated alpaca-style."""
import json
import random
from datasets import load_dataset

random.seed(0)
OUT = []

# 1) aya_dataset vie — instruction/response pairs (~8k)
try:
    aya = load_dataset("CohereForAI/aya_dataset", split="train")
    aya_vi = aya.filter(lambda x: x["language_code"] == "vie")
    print("aya vie:", len(aya_vi))
    for r in aya_vi:
        if len(r["inputs"]) < 2000 and len(r["targets"]) < 3000:
            OUT.append(("inst", r["inputs"].strip(), r["targets"].strip()))
except Exception as e:
    print("aya fail:", e)

# 2) Viet-OpenOrca subset — has vi column
try:
    orca = load_dataset("Open-Orca/Viet-OpenOrca-Subset-10K", split="train")
    print("orca:", len(orca))
    for r in orca:
        q = r.get("question_vi", "") or r.get("question", "")
        a = r.get("response_vi", "") or r.get("response", "")
        if q and a and len(q) < 2000 and len(a) < 3000:
            OUT.append(("inst", q.strip(), a.strip()))
except Exception as e:
    print("orca fail:", e)

# 3) ViVQA / Vistral-style — try Vistral 1.5B dataset (chat SFT data)
try:
    vistral = load_dataset("Vistral/sft_dataset", split="train")
    print("vistral:", len(vistral))
    for r in vistral:
        msgs = r.get("messages") or []
        if len(msgs) >= 2:
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if user and asst and len(user) < 2000 and len(asst) < 3000:
                OUT.append(("inst", user.strip(), asst.strip()))
except Exception as e:
    print("vistral fail:", e)

random.shuffle(OUT)
print("total pairs:", len(OUT))
n_val = max(200, int(len(OUT) * 0.02))


def emit(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for kind, q, a in rows:
            t = "### Instruction:\n" + q + "\n\n### Response:\n" + a
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


emit("sft_vi.jsonl", OUT[n_val:])
emit("sft_val.jsonl", OUT[:n_val])
print("train:", len(OUT) - n_val, "val:", n_val)
