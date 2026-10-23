"""
BridgeSign – Sentence brain
===========================
Turns a sequence of recognised sign glosses (ME HEAD HURT) into candidate
English sentences ("My head hurts.") with a small local T5 model exported to
ONNX by tools/sentence_brain/train.py. Runs offline on CPU; no text leaves
the server.

The brain only suggests: the signer confirms a sentence before it is shown
or spoken to the other person. Candidates that lose a finger-spelled name
are dropped, and when the model is unavailable a literal reading of the
signs is returned instead of nothing.
"""

import json
import os
import re
import threading

import numpy as np

import config
import sentence_vocab

MODEL_DIR = os.path.join(config.MODELS_DIR, "sentence_brain")
LOW_CONFIDENCE = 0.6     # signs below this also try their recognizer alternatives
MAX_VARIANTS = 3         # gloss sequences decoded per request
BEAM_SIZE = 4
MIN_SHARE = 0.15         # hide alternatives the model gives under 15% of the shown preference
# Spelled names are swapped for names the model saw often in training, then
# swapped back: it copies these reliably but mangles rare spellings.
NAME_STANDINS = ["John", "Mary", "Peter", "Grace", "James", "Ruth"]


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class SentenceBrain:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.meta = {}
        self.encoder = None
        self.decoder = None
        self.tokenizer = None
        self.last_error = None
        self._lock = threading.Lock()
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────

    def _load(self):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            with open(os.path.join(self.model_dir, "meta.json"), encoding="utf-8") as f:
                self.meta = json.load(f)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.encoder = ort.InferenceSession(
                os.path.join(self.model_dir, "encoder.onnx"), options, providers=["CPUExecutionProvider"]
            )
            self.decoder = ort.InferenceSession(
                os.path.join(self.model_dir, "decoder.onnx"), options, providers=["CPUExecutionProvider"]
            )
            self.tokenizer = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
            print(f"[SentenceBrain] Loaded local model from {self.model_dir}")
        except Exception as error:  # missing files, LFS pointers, runtime errors
            self.encoder = self.decoder = self.tokenizer = None
            self.last_error = str(error)
            print(f"[SentenceBrain] Model unavailable ({error}); using literal readings.")

    def is_available(self):
        return self.encoder is not None and self.decoder is not None and self.tokenizer is not None

    # ── Public API ────────────────────────────────────────────────────────

    def suggest(self, signs, k=3):
        """
        Parameters
        ----------
        signs : list of str or dict
            Each item is a gloss ("HEAD") or {"gloss": "HEAD", "confidence": 0.7,
            "alternatives": [{"gloss": "HELP", "confidence": 0.2}]}.

        Returns
        -------
        list of {"text", "score", "glosses", "source"} – best first, at most k.
        """
        items = [self._normalize_item(item) for item in signs or []]
        items = [item for item in items if item["gloss"]]
        if not items:
            return []

        variants = self._variants(items)
        if not self.is_available():
            return [self._literal(variants[0][0])]

        candidates = {}
        for glosses, variant_logp in variants:
            model_glosses, names = self._stand_in_names(glosses)
            for text, logp in self._generate(model_glosses):
                text = self._clean(self._restore_names(text, names))
                if not text or not self._keeps_names(glosses, text):
                    continue
                key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
                score = logp + variant_logp
                if key not in candidates or score > candidates[key]["score"]:
                    candidates[key] = {"text": text, "score": score, "glosses": glosses, "source": "local-model"}
        ranked = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)[:k]
        if not ranked:
            return [self._literal(variants[0][0])]
        # Report a relative preference that sums to 1 across shown candidates,
        # dropping alternatives too unlikely to be worth the signer's attention.
        top = ranked[0]["score"]
        weights = [float(np.exp(c["score"] - top)) for c in ranked]
        total = sum(weights)
        kept = [c for c, w in zip(ranked, weights) if w / total >= MIN_SHARE or c is ranked[0]]
        weights = [w for c, w in zip(ranked, weights) if c in kept]
        total = sum(weights)
        for cand, weight in zip(kept, weights):
            cand["score"] = round(weight / total, 3)
        return kept

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_item(item):
        if isinstance(item, dict):
            gloss = sentence_vocab.normalize_gloss(item.get("gloss"))
            conf = float(item.get("confidence", 1.0) or 1.0)
            alts = [
                (sentence_vocab.normalize_gloss(a.get("gloss")), float(a.get("confidence", 0.0) or 0.0))
                for a in item.get("alternatives") or []
                if isinstance(a, dict)
            ]
        else:
            gloss, conf, alts = sentence_vocab.normalize_gloss(item), 1.0, []
        return {"gloss": gloss, "confidence": max(min(conf, 1.0), 1e-3), "alternatives": [a for a in alts if a[0]]}

    @staticmethod
    def _variants(items):
        """The best gloss sequence plus single swaps for uncertain signs."""
        best = [item["gloss"] for item in items]
        variants = [(best, 0.0)]
        swaps = []
        for i, item in enumerate(items):
            if item["confidence"] >= LOW_CONFIDENCE:
                continue
            for alt, conf in item["alternatives"]:
                if alt != item["gloss"] and conf > 0:
                    swaps.append((i, alt, float(np.log(max(conf, 1e-3)) - np.log(item["confidence"]))))
        for i, alt, delta in sorted(swaps, key=lambda s: s[2], reverse=True)[: MAX_VARIANTS - 1]:
            seq = list(best)
            seq[i] = alt
            variants.append((seq, delta))
        return variants

    @staticmethod
    def _stand_in_names(glosses):
        """Replace spelled words with stand-in names; returns (glosses, {stand-in: real})."""
        prefix = sentence_vocab.FINGERSPELL_PREFIX
        real = [g[len(prefix):].capitalize() for g in glosses if g.startswith(prefix)]
        free = [n for n in NAME_STANDINS if n not in real]
        mapping, out = {}, []
        for gloss in glosses:
            if gloss.startswith(prefix) and free:
                name = gloss[len(prefix):].capitalize()
                standin = next((s for s, r in mapping.items() if r == name), None) or free.pop(0)
                mapping[standin] = name
                out.append(prefix + standin.upper())
            else:
                out.append(gloss)
        return out, mapping

    @staticmethod
    def _restore_names(text, mapping):
        for standin, name in mapping.items():
            text = re.sub(rf"\b{re.escape(standin)}\b", name, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _clean(text):
        """Drop garbled output and tidy punctuation."""
        if "�" in text or not re.search(r"[A-Za-z]", text):
            return ""
        text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        return text.strip()

    @staticmethod
    def _keeps_names(glosses, text):
        lower = text.lower()
        for gloss in glosses:
            if gloss.startswith(sentence_vocab.FINGERSPELL_PREFIX):
                name = gloss[len(sentence_vocab.FINGERSPELL_PREFIX):].lower()
                if name and name not in lower:
                    return False
        return True

    @staticmethod
    def _literal(glosses):
        words = []
        for gloss in glosses:
            if gloss == "Q":
                continue
            if gloss.startswith(sentence_vocab.FINGERSPELL_PREFIX):
                words.append(gloss[len(sentence_vocab.FINGERSPELL_PREFIX):].capitalize())
            else:
                words.append(gloss.replace("-", " ").lower())
        text = " ".join(words)
        text = text[:1].upper() + text[1:] + ("?" if "Q" in glosses else ".")
        return {"text": text, "score": 1.0, "glosses": glosses, "source": "literal"}

    def _generate(self, glosses):
        prefix = self.meta.get("task_prefix", "gloss: ")
        max_src = int(self.meta.get("max_source_tokens", 48))
        max_tgt = int(self.meta.get("max_target_tokens", 48))
        eos = int(self.meta.get("eos_token_id", 1))
        start = int(self.meta.get("decoder_start_token_id", 0))

        ids = self.tokenizer.encode(prefix + " ".join(glosses)).ids[:max_src]
        input_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(input_ids)
        with self._lock:
            hidden = self.encoder.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]

            beams = [([start], 0.0)]
            finished = []
            for _ in range(max_tgt):
                seqs = np.array([b[0] for b in beams], dtype=np.int64)
                n = len(beams)
                logits = self.decoder.run(None, {
                    "decoder_input_ids": seqs,
                    "encoder_hidden_states": np.repeat(hidden, n, axis=0),
                    "attention_mask": np.repeat(mask, n, axis=0),
                })[0][:, -1, :]
                logp = _log_softmax(logits.astype(np.float32))
                expanded = []
                for b, (seq, score) in enumerate(beams):
                    top = np.argpartition(-logp[b], BEAM_SIZE)[:BEAM_SIZE]
                    for token in top:
                        expanded.append((seq + [int(token)], score + float(logp[b, token])))
                expanded.sort(key=lambda e: e[1], reverse=True)
                beams = []
                for seq, score in expanded:
                    if seq[-1] == eos:
                        finished.append((seq, score / (len(seq) - 1) ** 0.6))
                    else:
                        beams.append((seq, score))
                    if len(beams) == BEAM_SIZE:
                        break
                if not beams or len(finished) >= BEAM_SIZE:
                    break
        if not finished:
            finished = [(seq, score / max(len(seq) - 1, 1) ** 0.6) for seq, score in beams]

        out = []
        seen = set()
        for seq, score in sorted(finished, key=lambda f: f[1], reverse=True):
            text = self.tokenizer.decode([t for t in seq[1:] if t != eos], skip_special_tokens=True).strip()
            if text and text not in seen:
                seen.add(text)
                out.append((text, score))
        return out


_brain = None
_brain_lock = threading.Lock()


def get_brain():
    """Load the brain once per process."""
    global _brain
    with _brain_lock:
        if _brain is None:
            _brain = SentenceBrain()
        return _brain
