"""Stream wikipedia-vi once. Write train/val jsonl + heldout txts.
Retried iteration: CI runners hit transient HF errors mid-stream.
Single source of truth is the kept list; retries resume past it."""
import json
import os
import time

from datasets import load_dataset

TRAIN_CHARS = 200_000_000  # ~50M tokens, pilot only
HELDOUT_ARTICLES = 500
EN_HELDOUT = 200
MIN_LEN = 500
MAX_RETRIES = 8


def open_stream(lang):
    kw = {"split": "train", "streaming": True}
    if os.environ.get("HF_TOKEN"):
        kw["token"] = os.environ["HF_TOKEN"]
    try:
        return load_dataset("wikimedia/wikipedia", f"20231101.{lang}", **kw)
    except Exception:
        return load_dataset("wikipedia", f"20231101.{lang}", **kw)


def collect(lang, need_chars=None, need_articles=None, skip_chars=0):
    """Collect article texts. Retries reopen the stream and fast-forward
    past skip_chars + everything already kept, so no duplicates, no loss."""
    texts, kept_chars = [], 0
    for attempt in range(MAX_RETRIES + 1):
        try:
            to_skip, skipped = skip_chars + kept_chars, 0
            for row in open_stream(lang):
                text = row["text"].strip()
                if len(text) < MIN_LEN:
                    continue
                if skipped < to_skip:
                    skipped += len(text)
                    continue
                texts.append(text)
                kept_chars += len(text)
                if need_chars and kept_chars >= need_chars:
                    return texts
                if need_articles and len(texts) >= need_articles:
                    return texts
            return texts  # stream exhausted
        except Exception as e:
            if attempt >= MAX_RETRIES:
                print(f"gave up after {MAX_RETRIES} retries, kept {len(texts)}",
                      flush=True)
                return texts
            wait = min(2 ** attempt, 60)
            print(f"stream blip ({e}) — kept {len(texts)} articles, "
                  f"retry in {wait}s", flush=True)
            time.sleep(wait)


def dump_jsonl(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def main():
    rows = collect("vi", need_chars=TRAIN_CHARS)
    train_chars = sum(len(t) for t in rows)
    heldout = collect("vi", need_articles=HELDOUT_ARTICLES,
                      skip_chars=train_chars)

    n_val = max(50, int(len(rows) * 0.02))
    dump_jsonl("train_vi.jsonl", rows[:-n_val])
    dump_jsonl("val_vi.jsonl", rows[-n_val:])
    with open("heldout_vi.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(heldout))
    print(f"train chars: {train_chars}, heldout: {len(heldout)}", flush=True)

    en = collect("en", need_articles=EN_HELDOUT)
    with open("heldout_en.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(en))
    print(f"en heldout: {len(en)}", flush=True)


if __name__ == "__main__":
    main()
