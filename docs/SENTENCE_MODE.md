# Sentence Mode and the Sentence Brain

**Branch:** `version-4` · **Status:** proof of concept for testing with the team.

Sentence mode lets a signer and another person (anyone, not only a clinician)
hold a two-way conversation. The signer signs words; a small local model turns
them into English sentence suggestions; the signer taps the one they meant and
it is shown in large text and spoken aloud. The other person replies by voice,
typing, or quick replies, and their words appear in large text for the signer.

## How it works

```
Camera ──► hand landmarks ──► gesture model (words) / letter model (fingerspelling)
                                   │
                                   ▼
                    signed glosses  e.g.  ME  HEAD  HURT
                                   │   (+ taps / typed words / "?" toggle)
                                   ▼
          Sentence brain (local T5-small, ONNX, CPU, offline)
                                   │
                                   ▼
          "My head hurts." | "I have a headache." | …   ← signer taps one
                                   │
                                   ▼
          Conversation log (large text) + text-to-speech for the other person

Other person ──► microphone (browser speech-to-text) / typing / quick replies
             ──► conversation log (large text) for the signer
```

- **Nothing is sent until the signer confirms.** Suggestions are only
  suggestions; tapping one is the signer saying it.
- **Offline.** The brain runs on the app server with `onnxruntime`; no text is
  sent to an outside AI service. (Browser speech-to-text for the other person
  uses the browser's own speech service.)
- **Uncertain signs.** The brain accepts per-sign alternatives with
  confidences and also tries the alternative for a sign below 60% confidence.
- **Names are protected.** Fingerspelled words (`fs-JOHN`) must appear in a
  suggestion, otherwise that suggestion is dropped.
- **No model, no blank screen.** If the model files are missing, the page shows
  a literal reading of the signs ("Me head hurt.") and says so.

### Using it

1. Live tab → **Start Camera** → choose **Sentence**.
2. Sign words (the 19 trained word gestures) or fingerspell any word; drop
   your hand briefly between fingerspelled words. Signs collect in
   **Signed words**. Remove a wrong sign with ×.
3. Use **Add sign** to tap signs from the everyday vocabulary (144 signs in
   groups) or type a word or name — useful for signs the camera does not
   recognise yet.
4. Turn on **? Question** for yes/no questions (eyebrows-up questions cannot be
   seen by the recognizer).
5. Tap the sentence you meant. It appears in the conversation and is spoken.
6. The other person taps **Speak**, types, or uses a quick reply.

## Files

| File | Purpose |
|---|---|
| `sentence_vocab.py` | The 144 everyday glosses, grouped; maps recognizer labels and typed words to glosses. |
| `sentence_brain.py` | Runtime: loads the ONNX model, beam search, alternatives, name check, literal fallback. |
| `tools/sentence_brain/build_pairs.py` | Builds training pairs: generated everyday pairs + short ASLG-PC12 pairs. |
| `tools/sentence_brain/train.py` | Fine-tunes T5-small, evaluates, exports `models/sentence_brain/` (ONNX, int8). |
| `tools/sentence_brain/probe.py` | Prints suggestions for hand-written sign sequences. |
| `static/sentence.js` | Sentence-mode page: sign tray, picker, suggestions, conversation. |
| `/api/sentence/vocab`, `/api/sentence/suggest`, `/api/sentence/clear` | Server endpoints. |

## Training data

- **Generated everyday pairs** (`build_pairs.py`): a small English grammar over
  the vocabulary — greetings, needs, feelings, pain, questions, past/future
  time signs, negation, names and places by fingerspelling, requests,
  emergencies. Glosses use sign-language order (time first, WH-word last,
  flexible subject position). Only meaning-preserving noise is added (a
  repeated sign); signs are never dropped, so the model is not taught to
  invent words.
- **ASLG-PC12** (Othman & Jemni, 2012; Hugging Face `achrafothman/aslg_pc12`):
  English/ASL-gloss pairs from European Parliament text, licensed
  **CC BY-NC 4.0**. Short pairs are used under a separate `gloss formal:`
  prefix so their style does not leak into the app's output.
  **Because of this licence the trained model is for non-commercial use.**
  Retrain without ASLG-PC12 (`build_pairs.py --aslg 0`) before any commercial use.

Rebuild and retrain (about 75 minutes on a 4-core laptop CPU):

```bash
pip install -r requirements-train.txt
python tools/sentence_brain/build_pairs.py --aslg 6000
python tools/sentence_brain/train.py
python tools/sentence_brain/probe.py
```

## Results

First model, trained 2026-10-23 on a 4-core laptop CPU (Intel i5-1135G7,
2 epochs, 1,930 steps, about 2 hours with the machine also in use):

| Measure | Result |
|---|---|
| Training pairs | 30,880 (24,880 generated everyday + 6,000 ASLG-PC12) |
| Held-out everyday pairs, exact sentence match | **98.6%** (1,382 pairs) |
| Model size (int8 ONNX) | 94 MB (encoder 35.5 MB + decoder 58.5 MB) |
| Suggestion time on the same CPU | 20–125 ms per request |

The held-out pairs come from the same generator as the training pairs, so
98.6% shows the patterns were learned — not how well it handles real signers.

Hand-written probes (`tools/sentence_brain/probe.py`), top suggestion:

| Signs | Suggestion |
|---|---|
| `ME HEAD HURT` / `HEAD HURT ME` | My head hurts. |
| `ME FEEL NOT GOOD` | I don't feel well. |
| `YESTERDAY ME GO HOSPITAL` | I went to the hospital yesterday. |
| `YOU NAME WHAT` | What is your name? |
| `MY FATHER NAME fs-BUKAMA` | My father's name is Bukama. |
| `ME LIVE fs-ZOMBA` | I live in Zomba. |
| `ME DEAF PLEASE WRITE` | I'm deaf. Please write it down. |
| `ME WANT GO HOME` *(not a generator pattern)* | I want to go home. |
| `DOCTOR WHERE` *(not a generator pattern)* | Where is the doctor? |
| `FIRE HELP` *(not a generator pattern)* | There is a fire. Help! |
| `ME TIRED WANT SLEEP` *(not a generator pattern)* | I'm tired. Want a nap. / I'm tired. Want to sleep? |
| `YOU HELP ME FIND TOILET` *(not a generator pattern)* | Can you help me? I'm finding a toilet. |

Known errors seen in testing:

- **Negation can spread across two sentences**: `ME UNDERSTAND ME KNOW NOT` →
  "I don't understand. I don't know." This changes meaning — the signer must
  check before sending.
- **Unseen combinations get clumsy**: "I'm finding a toilet", "Want a nap".
- **A trailing NO may be read as negation** of the previous sentence.

Fixes already applied at runtime: rare spelled names are swapped for a
stand-in name the model copies reliably and restored afterwards (fixes
"Bukama" → "Bukamara"); alternatives under 15% preference, garbled characters
and duplicate punctuation are hidden. The generator's WHEN questions are now
always future tense ("When will…"); the first model predates this fix.

### Retraining on a faster PC

Training is not tied to one machine. With an NVIDIA GPU it takes roughly
10 minutes:

```bash
# CUDA PyTorch first (pick the command for your CUDA version at pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
python tools/sentence_brain/build_pairs.py --aslg 6000   # downloads ASLG-PC12 if missing
python tools/sentence_brain/train.py                     # uses the GPU automatically
python tools/sentence_brain/probe.py
```

`train.py --init-from data/sentence_brain/checkpoint` continues from an earlier
checkpoint instead of the base T5-small (useful after adding vocabulary).

## Limits — read before testing with others

- **Recognition is the bottleneck.** The camera currently recognises 19 word
  gestures plus fingerspelling. Most of the 144 brain signs must be tapped or
  typed until more signs are recorded and the gesture model is retrained.
- **Generated, not signed, sentences.** The brain learned from generated
  gloss/English pairs. Real signers use other orders, classifiers, and facial
  grammar it has never seen. Expect misses outside everyday patterns.
- **ASL-style glosses.** Gloss order follows common ASL patterns. MSL order and
  vocabulary need review by Deaf MSL signers; the gloss layer makes it possible
  to add an MSL vocabulary and pairs later.
- **No context yet.** Each request is translated on its own; the other
  person's last message does not influence suggestions.
- **Not an interpreter.** Do not rely on it for medical, legal, or emergency
  decisions. Keep typing, the phrase/quick-reply options, and a human
  interpreter route available.
