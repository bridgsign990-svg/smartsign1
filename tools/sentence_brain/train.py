"""
Fine-tune T5-small on gloss -> English pairs and export the sentence brain.

Training needs torch + transformers (see requirements-train.txt); the app
only needs onnxruntime + tokenizers at runtime.

    python tools/sentence_brain/build_pairs.py
    python tools/sentence_brain/train.py                 # train, evaluate, export
    python tools/sentence_brain/train.py --export-only   # re-export a saved checkpoint

Uses an NVIDIA GPU automatically when CUDA PyTorch is installed (about 10 minutes
instead of about 75 on a laptop CPU). Outputs
models/sentence_brain/{encoder.onnx, decoder.onnx, tokenizer.json, meta.json}.
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import date
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE_MODEL = "google-t5/t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SRC = 48
MAX_TGT = 48


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def batches(rows, size, shuffle, seed=0):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), size):
        yield [rows[j] for j in order[i:i + size]]


def encode(tokenizer, batch):
    enc = tokenizer([r["src"] for r in batch], max_length=MAX_SRC, truncation=True, padding=True, return_tensors="pt")
    labels = tokenizer([r["tgt"] for r in batch], max_length=MAX_TGT, truncation=True, padding=True,
                       return_tensors="pt").input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return enc.to(DEVICE), labels.to(DEVICE)


@torch.no_grad()
def evaluate(model, tokenizer, rows, limit=None, batch_size=64):
    model.eval()
    rows = rows[:limit] if limit else rows
    exact = 0
    misses = []
    for batch in batches(rows, batch_size, shuffle=False):
        enc = tokenizer([r["src"] for r in batch], max_length=MAX_SRC, truncation=True, padding=True,
                        return_tensors="pt").to(DEVICE)
        out = model.generate(**enc, max_new_tokens=MAX_TGT, num_beams=1)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        for row, pred in zip(batch, preds):
            if pred.strip() == row["tgt"].strip():
                exact += 1
            elif len(misses) < 12:
                misses.append((row["src"], row["tgt"], pred))
    model.train()
    return exact / max(len(rows), 1), misses


def train(args):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    data = args.data
    train_rows = load_jsonl(data / "train.jsonl")
    val_rows = load_jsonl(data / "val.jsonl")

    start_from = str(args.init_from) if args.init_from else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(start_from)
    model = T5ForConditionalGeneration.from_pretrained(start_from, use_safetensors=True).to(DEVICE)
    model.train()

    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(1, int(0.03 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.05, 1 - s / total_steps)
    )

    ckpt = args.checkpoint
    best = -1.0
    step = 0
    started = time.time()
    print(f"device={DEVICE} start={start_from} train={len(train_rows)} val={len(val_rows)} steps={total_steps}",
          flush=True)
    for epoch in range(args.epochs):
        for batch in batches(train_rows, args.batch_size, shuffle=True, seed=args.seed + epoch):
            enc, labels = encode(tokenizer, batch)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 50 == 0:
                rate = step / (time.time() - started)
                eta = (total_steps - step) / max(rate, 1e-9) / 60
                print(f"step {step}/{total_steps} loss {loss.item():.3f} ({rate:.2f} it/s, ~{eta:.0f} min left)", flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                acc, _ = evaluate(model, tokenizer, val_rows, limit=400)
                print(f"  val exact-match (400): {acc:.3f}", flush=True)
                if acc > best:
                    best = acc
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"  saved checkpoint ({best:.3f})", flush=True)
    return best


def report(args):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).to(DEVICE)
    test = load_jsonl(args.data / "test.jsonl")
    acc, misses = evaluate(model, tokenizer, test)
    print(f"test exact-match: {acc:.3f} on {len(test)} held-out everyday pairs")
    for src, tgt, pred in misses:
        print(f"  {src!r}\n    want {tgt!r}\n    got  {pred!r}")
    return acc, len(test)


class _Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class _Decoder(torch.nn.Module):
    """One full decoder pass (no KV cache): sentences are short, so this stays fast and simple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, encoder_hidden_states, attention_mask):
        out = self.model(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def export(args, test_accuracy=None, test_size=None):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint).eval()

    ids = tokenizer(["gloss: ME HEAD HURT"], return_tensors="pt")
    with torch.no_grad():
        hidden = _Encoder(model)(ids.input_ids, ids.attention_mask)
    dec_ids = torch.tensor([[model.config.decoder_start_token_id]])

    tmp = out / "_fp32"
    tmp.mkdir(exist_ok=True)
    torch.onnx.export(
        _Encoder(model), (ids.input_ids, ids.attention_mask), tmp / "encoder.onnx",
        input_names=["input_ids", "attention_mask"], output_names=["hidden"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "hidden": {0: "b", 1: "s"}},
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        _Decoder(model), (dec_ids, hidden, ids.attention_mask), tmp / "decoder.onnx",
        input_names=["decoder_input_ids", "encoder_hidden_states", "attention_mask"], output_names=["logits"],
        dynamic_axes={"decoder_input_ids": {0: "b", 1: "t"}, "encoder_hidden_states": {0: "b", 1: "s"},
                      "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b", 1: "t"}},
        opset_version=17, dynamo=False,
    )
    for name in ("encoder.onnx", "decoder.onnx"):
        quantize_dynamic(str(tmp / name), str(out / name), weight_type=QuantType.QInt8)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()

    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    meta = {
        "base_model": BASE_MODEL,
        "task_prefix": "gloss: ",
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "decoder_start_token_id": model.config.decoder_start_token_id,
        "max_source_tokens": MAX_SRC,
        "max_target_tokens": MAX_TGT,
        "trained": date.today().isoformat(),
        "training_data": {
            "conversation": "tools/sentence_brain/build_pairs.py (generated everyday gloss/English pairs)",
            "formal": "ASLG-PC12 (Othman & Jemni 2012), CC BY-NC 4.0, short pairs, 'gloss formal:' prefix",
        },
        "evaluation": {
            "test_exact_match": test_accuracy,
            "test_pairs": test_size,
            "note": "Held-out generated pairs; measures grammar coverage, not real signer conversations.",
        },
        "license_note": "Trained partly on ASLG-PC12 (CC BY-NC 4.0): non-commercial use only.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in out.iterdir()}
    print(f"exported to {out}: {sizes} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sentence_brain")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "sentence_brain" / "checkpoint")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "sentence_brain")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps-probe", type=int, default=0, help="time a few steps and exit")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="continue from a saved checkpoint directory instead of the base T5-small")
    args = parser.parse_args()

    if args.max_steps_probe:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL, use_safetensors=True)
        torch.set_num_threads(args.threads)
        rows = load_jsonl(args.data / "train.jsonl")[: args.batch_size * args.max_steps_probe]
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        t = time.time()
        for batch in batches(rows, args.batch_size, shuffle=False):
            enc, labels = encode(tok, batch)
            model(**enc, labels=labels).loss.backward()
            opt.step()
            opt.zero_grad()
        per = (time.time() - t) / args.max_steps_probe
        steps = math.ceil(len(load_jsonl(args.data / "train.jsonl")) / args.batch_size) * args.epochs
        print(f"{per:.2f} s/step -> ~{per * steps / 60:.0f} min for {args.epochs} epochs ({steps} steps)")
        return

    if not args.export_only:
        train(args)
    acc, n = report(args)
    export(args, acc, n)


if __name__ == "__main__":
    main()
