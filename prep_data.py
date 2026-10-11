"""Stream wikipedia-vi once. Write train.jsonl (pilot cap) + heldout.txt.
Run locally, upload both to Vast box. Never train on heldout."""
from datasets import load_dataset

TRAIN_CHARS = 200_000_000  # ~50M tokens, pilot only
HELDOUT_ARTICLES = 500
MIN_LEN = 500

try:
    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
except Exception:
    ds = load_dataset("wikipedia", "20231101.vi", split="train", streaming=True)  # legacy id

train_chars, heldout, rows = 0, [], []
for row in ds:
    text = row["text"].strip()
    if len(text) < MIN_LEN:
        continue
    if train_chars < TRAIN_CHARS:
        rows.append({"text": text})
        train_chars += len(text)
    elif len(heldout) < HELDOUT_ARTICLES:
        heldout.append(text)
    else:
        break

import json
VAL_FRAC = 0.02  # last 2% of train rows double as val; heldout untouched for final gate
n_val = max(50, int(len(rows) * VAL_FRAC))
val_rows, train_rows = rows[-n_val:], rows[:-n_val]
with open("train_vi.jsonl", "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("val_vi.jsonl", "w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("heldout_vi.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(heldout))
print(f"train chars: {train_chars}, heldout articles: {len(heldout)}")

# English sanity heldout (200 articles, never trained on)
try:
    en_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
except Exception:
    en_ds = load_dataset("wikipedia", "20231101.en", split="train", streaming=True)
en_heldout = []
for row in en_ds:
    text = row["text"].strip()
    if len(text) < MIN_LEN:
        continue
    en_heldout.append(text)
    if len(en_heldout) >= 200:
        break
with open("heldout_en.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(en_heldout))
print(f"en heldout articles: {len(en_heldout)}")
