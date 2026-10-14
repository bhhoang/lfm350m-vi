"""Eval v2 — repetition penalties so outputs don't loop."""
import sys
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from peft import PeftModel

BASE = "lfm350m-vi-cpt-merged"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "lfm350m-vi-sft-v2"

PROMPTS = [
    "Giải thích ngắn gọn nguyên nhân gây ra mưa axit.",
    "Viết một câu chúc mừng sinh nhật cho người bạn thân.",
    "Liệt kê ba món ăn đặc trưng của miền Bắc Việt Nam.",
    "Dịch sang tiếng Anh: 'Hà Nội là thủ đô của Việt Nam.'",
    "Thủ đô của Pháp là gì? Trả lời bằng tiếng Việt.",
    "Viết một đoạn mã Python in ra các số chẵn từ 0 đến 10.",
    "Tóm tắt trong một câu: Quang hợp là quá trình thực vật dùng ánh sáng mặt trời để tạo năng lượng.",
    "Ai viết tác phẩm Truyện Kiều?",
]

_inner = Tokenizer.from_pretrained("LiquidAI/LFM2.5-350M-Base")
t = PreTrainedTokenizerFast(tokenizer_object=_inner)
t.eos_token = "<|im_end|>"
t.pad_token = "<|pad|>"

m = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto")
m = PeftModel.from_pretrained(m, CKPT)
m = m.merge_and_unload()
m.eval()

for p in PROMPTS:
    s = "### Instruction:\n" + p + "\n\n### Response:\n"
    ids = t(s, return_tensors="pt").input_ids.to(m.device)
    with torch.no_grad():
        out = m.generate(
            ids, max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=t.pad_token_id,
            eos_token_id=t.eos_token_id)
    ans = t.decode(out[0][ids.size(1):], skip_special_tokens=True)
    print("P:", p, flush=True)
    print("A:", ans.split("###")[0].strip()[:400], flush=True)
    print("=" * 60, flush=True)
