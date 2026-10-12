"""CPT pilot, LoRA, LFM2.5-350M-Base, Unsloth. 1 GPU, bf16. Run on Vast box."""
import argparse
import torch
from datasets import load_dataset
try:  # transformers v5 moved things; duck-type fallback still works
    from transformers.trainer_callback import TrainerCallback
except ImportError:
    class TrainerCallback:
        pass
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

MODEL = "LiquidAI/LFM2.5-350M-Base"
MAX_SEQ = 2048


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
ap.add_argument("--max-steps", type=int, default=800)
ap.add_argument("--output-dir", default="lfm350m-vi-cpt-pilot")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--init-adapter", default=None,
                help="warm-start LoRA from an adapter dir instead of fresh")
cli = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(
    MODEL, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=False)
if cli.init_adapter:  # continue CPT: reuse the previous round's LoRA
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, cli.init_adapter, is_trainable=True)
else:
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth")

ds = load_dataset("json", data_files="train_vi.jsonl", split="train")
val_ds = load_dataset("json", data_files="val_vi.jsonl", split="train")
bf16 = torch.cuda.is_bf16_supported()
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, eval_dataset=val_ds, args=SFTConfig(
    dataset_text_field="text", per_device_train_batch_size=4,
    gradient_accumulation_steps=8, max_steps=cli.max_steps, learning_rate=cli.lr,
    fp16=not bf16, bf16=bf16, logging_steps=10, save_steps=400,
    output_dir=cli.output_dir,
    disable_tqdm=False,  # keep tqdm progress bar
    eval_strategy="epoch", per_device_eval_batch_size=4,
), callbacks=[EpochSeparatorCallback()])
trainer.train()
model.save_pretrained(cli.output_dir)
