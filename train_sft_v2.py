"""SFT v2 — bigger data, bigger LoRA (r=64), from merged CPT model."""
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass

MODEL = "lfm350m-vi-cpt-merged"
MAX_SEQ = 1024


class EpochSeparatorCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n===== Epoch {int(state.epoch) + 1} started =====", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n===== Epoch {int(state.epoch)} ended =====", flush=True)
        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                print(f"eval_loss: {entry['eval_loss']:.4f} at step {entry.get('step')}", flush=True)
                break


ap = argparse.ArgumentParser()
ap.add_argument("--max-steps", type=int, default=1500)
ap.add_argument("--output-dir", default="lfm350m-vi-sft-v2")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--model", default=MODEL)
cli = ap.parse_args()

_inner = Tokenizer.from_pretrained("LiquidAI/LFM2.5-350M-Base")
tok = PreTrainedTokenizerFast(tokenizer_object=_inner)
tok.eos_token = "<|im_end|>"
tok.pad_token = "<|pad|>"

model = AutoModelForCausalLM.from_pretrained(
    cli.model, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="sdpa")

model = get_peft_model(model, LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
model.train()

ds = load_dataset("json", data_files="sft_vi.jsonl", split="train")
val_ds = load_dataset("json", data_files="sft_val.jsonl", split="train")
bf16 = torch.cuda.is_bf16_supported()

trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, eval_dataset=val_ds, args=SFTConfig(
    dataset_text_field="text", per_device_train_batch_size=8,
    gradient_accumulation_steps=4, max_steps=cli.max_steps, learning_rate=cli.lr,
    fp16=not bf16, bf16=bf16, logging_steps=10, save_steps=300,
    output_dir=cli.output_dir,
    disable_tqdm=False,
    eval_strategy="epoch", per_device_eval_batch_size=8,
    gradient_checkpointing=False,
), callbacks=[EpochSeparatorCallback()])
trainer.train()
model.save_pretrained(cli.output_dir)
tok.save_pretrained(cli.output_dir)
