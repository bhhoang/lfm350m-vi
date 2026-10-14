"""Merge CPT LoRA into base, save as new model dir."""
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel
from tokenizers import Tokenizer

BASE = "LiquidAI/LFM2.5-350M-Base"
ADAPTER = "lfm350m-vi-cpt-r3"
OUT = "lfm350m-vi-cpt-merged"

m = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, ADAPTER)
m = m.merge_and_unload()
m.save_pretrained(OUT)
print("merged saved to", OUT)

t = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_pretrained(BASE))
t.eos_token = "<|im_end|>"
t.pad_token = "<|pad|>"
t.save_pretrained(OUT)
print("tokenizer saved")
