"""
Build gloss -> English training pairs for the sentence brain.

Two sources, each with its own T5 task prefix so their styles never mix:

- "gloss:"        everyday conversation pairs generated here from a small
                  English grammar over sentence_vocab.GLOSSES (the style the
                  app outputs: cased, punctuated, contractions).
- "gloss formal:" short ASLG-PC12 pairs (CC BY-NC 4.0, Othman & Jemni 2012),
                  normalised to the same pronoun glosses. They teach general
                  gloss-to-English grammar; the app never uses this prefix.

Glosses use sign-language order (time first, WH-word last, flexible subject
position). Only meaning-preserving noise is added (repeated signs, reordered
topic); signs are never dropped, so the model is not taught to invent words.

    python tools/sentence_brain/build_pairs.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sentence_vocab  # noqa: E402

PREFIX = "gloss: "
FORMAL_PREFIX = "gloss formal: "

# ── Lexicon ─────────────────────────────────────────────────────────────────

# gloss tokens -> (subject, object, possessive, person) ; person in {1s,2,3s,1p,3p}
SUBJECTS = {
    ("ME",): ("I", "me", "my", "1s"),
    ("YOU",): ("you", "you", "your", "2"),
    ("HE",): ("he", "him", "his", "3s"),
    ("SHE",): ("she", "her", "her", "3s"),
    ("WE",): ("we", "us", "our", "1p"),
    ("THEY",): ("they", "them", "their", "3p"),
    ("MY", "MOTHER"): ("my mother", "my mother", "my mother's", "3s"),
    ("MY", "FATHER"): ("my father", "my father", "my father's", "3s"),
    ("MY", "FRIEND"): ("my friend", "my friend", "my friend's", "3s"),
    ("MY", "CHILD"): ("my child", "my child", "my child's", "3s"),
    ("MY", "BABY"): ("my baby", "my baby", "my baby's", "3s"),
    ("DOCTOR",): ("the doctor", "the doctor", "the doctor's", "3s"),
    ("TEACHER",): ("the teacher", "the teacher", "the teacher's", "3s"),
}
PRONOUN_SUBJECTS = [("ME",), ("YOU",), ("HE",), ("SHE",), ("WE",), ("THEY",)]

BE_PRESENT = {"1s": "am", "2": "are", "3s": "is", "1p": "are", "3p": "are"}
BE_PAST = {"1s": "was", "2": "were", "3s": "was", "1p": "were", "3p": "were"}
CONTRACT = {"I am": "I'm", "you are": "you're", "he is": "he's", "she is": "she's",
            "we are": "we're", "they are": "they're"}

# gloss -> (base, 3s, past, ing, kind) ; kind: stative uses simple present without a time sign
VERBS = {
    "NEED": ("need", "needs", "needed", "needing", "stative"),
    "WANT": ("want", "wants", "wanted", "wanting", "stative"),
    "LIKE": ("like", "likes", "liked", "liking", "stative"),
    "HAVE": ("have", "has", "had", "having", "stative"),
    "KNOW": ("know", "knows", "knew", "knowing", "stative"),
    "UNDERSTAND": ("understand", "understands", "understood", "understanding", "stative"),
    "LIVE": ("live", "lives", "lived", "living", "stative"),
    "SEE": ("see", "sees", "saw", "seeing", "stative"),
    "GO": ("go", "goes", "went", "going", "action"),
    "COME": ("come", "comes", "came", "coming", "action"),
    "EAT": ("eat", "eats", "ate", "eating", "action"),
    "DRINK": ("drink", "drinks", "drank", "drinking", "action"),
    "HELP": ("help", "helps", "helped", "helping", "action"),
    "CALL": ("call", "calls", "called", "calling", "action"),
    "LOOK-FOR": ("look for", "looks for", "looked for", "looking for", "action"),
    "WORK": ("work", "works", "worked", "working", "action"),
    "LEARN": ("learn", "learns", "learned", "learning", "action"),
    "WAIT": ("wait", "waits", "waited", "waiting", "action"),
    "SLEEP": ("sleep", "sleeps", "slept", "sleeping", "action"),
    "BUY": ("buy", "buys", "bought", "buying", "action"),
    "PAY": ("pay", "pays", "paid", "paying", "action"),
    "FIND": ("find", "finds", "found", "finding", "action"),
    "LOSE": ("lose", "loses", "lost", "losing", "action"),
    "WRITE": ("write", "writes", "wrote", "writing", "action"),
}

# Things: gloss -> (indefinite phrase, definite phrase, plural)
THINGS = {
    "WATER": ("water", "the water", "water"),
    "FOOD": ("food", "the food", "food"),
    "MEDICINE": ("medicine", "the medicine", "medicine"),
    "MONEY": ("money", "the money", "money"),
    "BREAD": ("bread", "the bread", "bread"),
    "TEA": ("tea", "the tea", "tea"),
    "HELP": ("help", "the help", "help"),
    "PHONE": ("a phone", "the phone", "phones"),
    "BAG": ("a bag", "the bag", "bags"),
    "KEY": ("a key", "the key", "keys"),
    "BOOK": ("a book", "the book", "books"),
    "TICKET": ("a ticket", "the ticket", "tickets"),
    "TAXI": ("a taxi", "the taxi", "taxis"),
    "CAR": ("a car", "the car", "cars"),
    "DOCTOR": ("a doctor", "the doctor", "doctors"),
    "INTERPRETER": ("an interpreter", "the interpreter", "interpreters"),
    "AMBULANCE": ("an ambulance", "the ambulance", "ambulances"),
    "TEACHER": ("a teacher", "the teacher", "teachers"),
    "FRIEND": ("a friend", "the friend", "friends"),
    "TOILET": ("a toilet", "the toilet", "toilets"),
}
PERSONAL_THINGS = {"PHONE", "BAG", "KEY", "BOOK", "TICKET", "MONEY", "CAR"}

# Verb -> allowed object glosses
VERB_OBJECTS = {
    "NEED": ["WATER", "FOOD", "MEDICINE", "MONEY", "HELP", "DOCTOR", "INTERPRETER", "AMBULANCE", "TAXI", "TICKET", "PHONE"],
    "WANT": ["WATER", "FOOD", "TEA", "BREAD", "MEDICINE", "MONEY", "BOOK", "PHONE", "TAXI", "TICKET", "CAR"],
    "LIKE": ["TEA", "BREAD", "FOOD", "BOOK", "CAR", "PHONE", "WATER"],
    "HAVE": ["MONEY", "PHONE", "BAG", "KEY", "TICKET", "BOOK", "CAR", "MEDICINE", "FOOD", "WATER"],
    "BUY": ["FOOD", "BREAD", "TEA", "MEDICINE", "PHONE", "TICKET", "BOOK", "BAG", "WATER"],
    "FIND": ["PHONE", "BAG", "KEY", "MONEY", "BOOK", "TICKET", "DOCTOR", "TAXI"],
    "LOSE": ["PHONE", "BAG", "KEY", "MONEY", "BOOK", "TICKET"],
    "LOOK-FOR": ["PHONE", "BAG", "KEY", "MONEY", "BOOK", "DOCTOR", "TAXI", "TOILET", "FRIEND", "TEACHER"],
    "EAT": ["FOOD", "BREAD"],
    "DRINK": ["WATER", "TEA", "MEDICINE"],
    "WRITE": ["NAME", "BOOK"],
}
PERSON_VERBS = ["HELP", "CALL", "SEE", "UNDERSTAND", "KNOW", "WAIT"]  # take a person object
PERSON_OBJECTS = {
    ("ME",): "me", ("YOU",): "you", ("HE",): "him", ("SHE",): "her", ("US",): "us",
    ("THEM",): "them", ("MY", "MOTHER"): "my mother", ("MY", "FATHER"): "my father",
    ("MY", "FRIEND"): "my friend", ("DOCTOR",): "the doctor", ("TEACHER",): "the teacher",
    ("POLICE",): "the police", ("INTERPRETER",): "the interpreter",
}
PERSON_OBJECT_GLOSS = {("US",): ("WE",), ("THEM",): ("THEY",)}  # signs for us/them

# Places: gloss -> phrase after a motion verb
PLACES = {
    "HOME": "home", "SCHOOL": "to school", "HOSPITAL": "to the hospital", "CLINIC": "to the clinic",
    "CHURCH": "to church", "MARKET": "to the market", "SHOP": "to the shop", "BANK": "to the bank",
    "TOILET": "to the toilet", "TOWN": "to town", "PHARMACY": "to the pharmacy",
    "POLICE-STATION": "to the police station", "WORK": "to work",
}
PLACE_NOUN = {  # "where is ..."
    "HOSPITAL": "the hospital", "CLINIC": "the clinic", "CHURCH": "the church", "MARKET": "the market",
    "SHOP": "the shop", "BANK": "the bank", "TOILET": "the toilet", "PHARMACY": "the pharmacy",
    "POLICE-STATION": "the police station", "SCHOOL": "the school", "TAXI": "the taxi",
    "DOCTOR": "the doctor", "TEACHER": "the teacher", "INTERPRETER": "the interpreter",
}

ADJECTIVES = {
    "TIRED": "tired", "HUNGRY": "hungry", "THIRSTY": "thirsty", "SICK": "sick", "HAPPY": "happy",
    "SAD": "sad", "COLD": "cold", "HOT": "hot", "SCARED": "scared", "ANGRY": "angry", "FINE": "fine",
    "BUSY": "busy", "LOST": "lost", "LATE": "late", "READY": "ready", "WORRIED": "worried",
    "DIZZY": "dizzy", "DEAF": "deaf", "OK": "okay",
}
BODY = {
    "HEAD": "head", "STOMACH": "stomach", "CHEST": "chest", "BACK": "back", "LEG": "leg", "ARM": "arm",
    "EYE": "eye", "EAR": "ear", "TOOTH": "tooth", "THROAT": "throat", "HAND": "hand", "FOOT": "foot",
}

PAST_TIMES = {"YESTERDAY": "yesterday"}
FUTURE_TIMES = {"TOMORROW": "tomorrow", "LATER": "later"}
PRESENT_TIMES = {"NOW": "now", "TODAY": "today"}
HABIT_TIMES = {"EVERYDAY": "every day"}

FIXED = [
    (["HELLO"], ["Hello."]),
    (["HELLO", "HOW", "YOU"], ["Hello, how are you?"]),
    (["HOW", "YOU"], ["How are you?"]),
    (["YOU", "HOW"], ["How are you?"]),
    (["GOOD", "MORNING"], ["Good morning."]),
    (["GOOD", "AFTERNOON"], ["Good afternoon."]),
    (["GOOD", "NIGHT"], ["Good night."]),
    (["GOODBYE"], ["Goodbye."]),
    (["THANK-YOU"], ["Thank you."]),
    (["THANK-YOU", "VERY"], ["Thank you very much."]),
    (["WELCOME"], ["You're welcome."]),
    (["SORRY"], ["I'm sorry."]),
    (["ME", "SORRY"], ["I'm sorry."]),
    (["EXCUSE-ME"], ["Excuse me."]),
    (["YES"], ["Yes."]),
    (["NO"], ["No."]),
    (["OK"], ["Okay."]),
    (["PLEASE"], ["Please."]),
    (["NICE", "MEET", "YOU"], ["Nice to meet you."]),
    (["ME", "FINE"], ["I'm fine."]),
    (["ME", "FINE", "THANK-YOU"], ["I'm fine, thank you."]),
    (["ME", "DEAF"], ["I'm deaf."]),
    (["ME", "DEAF", "PLEASE", "WRITE"], ["I'm deaf. Please write it down."]),
    (["PLEASE", "WRITE"], ["Please write it down."]),
    (["WRITE", "PLEASE"], ["Please write it down."]),
    (["REPEAT", "PLEASE"], ["Please repeat that."]),
    (["PLEASE", "REPEAT"], ["Please repeat that."]),
    (["AGAIN", "PLEASE"], ["Please say that again."]),
    (["SLOW", "PLEASE"], ["Please slow down."]),
    (["PLEASE", "SLOW"], ["Please slow down."]),
    (["WAIT", "PLEASE"], ["Please wait."]),
    (["PLEASE", "WAIT"], ["Please wait."]),
    (["STOP"], ["Stop!"]),
    (["HELP"], ["Help!"]),
    (["HELP", "ME"], ["Help me!"]),
    (["HELP", "ME", "PLEASE"], ["Please help me."]),
    (["PLEASE", "HELP", "ME"], ["Please help me."]),
    (["YOU", "HELP", "ME", "PLEASE"], ["Can you help me, please?"]),
    (["YOU", "CAN", "HELP", "ME", "Q"], ["Can you help me?"]),
    (["EMERGENCY"], ["This is an emergency!"]),
    (["FIRE"], ["There is a fire!"]),
    (["DANGER"], ["Danger!"]),
    (["CALL", "AMBULANCE"], ["Call an ambulance!"]),
    (["AMBULANCE"], ["Call an ambulance!"]),
    (["CALL", "POLICE"], ["Call the police!"]),
    (["CALL", "DOCTOR"], ["Call a doctor!"]),
    (["CALL", "DOCTOR", "PLEASE"], ["Please call a doctor."]),
    (["ME", "NOT", "UNDERSTAND"], ["I don't understand."]),
    (["ME", "UNDERSTAND", "NOT"], ["I don't understand."]),
    (["ME", "UNDERSTAND"], ["I understand."]),
    (["YOU", "UNDERSTAND", "Q"], ["Do you understand?"]),
    (["ME", "NOT", "KNOW"], ["I don't know."]),
    (["ME", "KNOW", "NOT"], ["I don't know."]),
    (["YOU", "NAME", "WHAT"], ["What is your name?"]),
    (["WHAT", "YOU", "NAME"], ["What is your name?"]),
    (["NAME", "WHAT"], ["What is your name?"]),
    (["TIME", "WHAT"], ["What time is it?"]),
    (["WHAT", "TIME"], ["What time is it?"]),
    (["HOW-MUCH"], ["How much is it?"]),
    (["HOW-MUCH", "Q"], ["How much is it?"]),
    (["YOU", "FROM", "WHERE"], ["Where are you from?"]),
    (["WHERE", "YOU", "FROM"], ["Where are you from?"]),
    (["WHAT", "HAPPEN"], ["What happened?"]),
    (["WHY"], ["Why?"]),
    (["ME", "NEED", "HELP", "NOW"], ["I need help now."]),
    (["ME", "FEEL", "BAD"], ["I don't feel well."]),
    (["ME", "FEEL", "NOT", "GOOD"], ["I don't feel well."]),
    (["ME", "NOT", "FEEL", "GOOD"], ["I don't feel well."]),
    (["ME", "FEEL", "GOOD"], ["I feel good."]),
    (["ME", "FEEL", "SICK"], ["I feel sick."]),
    (["ME", "FEEL", "DIZZY"], ["I feel dizzy."]),
    (["YOU", "FEEL", "HOW"], ["How are you feeling?"]),
    (["HOW", "YOU", "FEEL"], ["How are you feeling?"]),
    (["ME", "HURT"], ["I'm hurt."]),
    (["WHERE", "HURT"], ["Where does it hurt?"]),
    (["HURT", "WHERE"], ["Where does it hurt?"]),
    (["ME", "NEED", "INTERPRETER"], ["I need an interpreter."]),
    (["FOR", "ME", "Q"], ["Is it for me?"]),
]

FIRST_NAMES = """John Mary Grace Peter James Esther Daniel Ruth Joseph Faith Chikondi Tiwonge Kondwani Chisomo
Mphatso Thoko Limbani Dalitso Takondwa Madalitso Yamikani Atupele Wezi Chimwemwe Pemphero Mwayi Kumbukani
Luka Sarah Ahmed Fatima Amina Musa Tendai Lindiwe Themba Zodwa Kofi Ama Ngozi Emeka Ali Hassan Lucy Brian
Alice Michael Benson Moses Aaron Jabulani Tamanda Chifundo Mercy Gift Blessings Precious Hope Patience""".split()
TOWNS = """Lilongwe Blantyre Zomba Mzuzu Kasungu Mangochi Salima Karonga Dedza Nkhotakota Balaka Liwonde
Nairobi Lusaka Harare Johannesburg London""".split()


def _syllable_name(rng):
    syllables = ["ka", "ti", "mo", "la", "ne", "chi", "wa", "zu", "pe", "ro", "ma", "di", "so", "bu", "ya", "le"]
    return "".join(rng.choice(syllables) for _ in range(rng.randint(2, 3))).capitalize()


def _name(rng, pool):
    return rng.choice(pool) if rng.random() < 0.8 else _syllable_name(rng)


def _fs(word):
    return sentence_vocab.FINGERSPELL_PREFIX + word.upper()


# ── English helpers ─────────────────────────────────────────────────────────

def _cap(sentence):
    return sentence[:1].upper() + sentence[1:]


def _contract(text):
    for full, short in CONTRACT.items():
        if text.startswith(full + " ") or text == full:
            return short + text[len(full):]
    return text


def _do(person, past=False):
    if past:
        return "did"
    return "does" if person == "3s" else "do"


def _clause(subj, person, verb, rest, tense, negative, question):
    """Realise subject + verb (+ rest) in a tense; returns English without final punctuation."""
    base, s3, past, ing, kind = VERBS[verb]
    rest = (" " + rest) if rest else ""
    if tense == "past":
        if question:
            return f"did {subj} {base}{rest}"
        return f"{subj} didn't {base}{rest}" if negative else f"{subj} {past}{rest}"
    if tense == "future":
        if question:
            return f"will {subj} {base}{rest}"
        return f"{subj} won't {base}{rest}" if negative else f"{subj} will {base}{rest}"
    if tense == "progressive" and kind == "action":
        be = BE_PRESENT[person]
        if question:
            return f"{be} {subj} {ing}{rest}"
        core = f"{subj} {be} not {ing}{rest}" if negative else f"{subj} {be} {ing}{rest}"
        return _contract(core)
    # simple present (stative, habitual)
    if question:
        return f"{_do(person)} {subj} {base}{rest}"
    if negative:
        return f"{subj} {'doesn' if person == '3s' else 'don'}'t {base}{rest}"
    return f"{subj} {s3 if person == '3s' else base}{rest}"


def _time(rng, allow_future=True, allow_past=True):
    """Pick a time sign; returns (gloss list, english suffix, tense)."""
    options = [("none", None)]
    if allow_past:
        options.append(("past", rng.choice(list(PAST_TIMES))))
    if allow_future:
        options.append(("future", rng.choice(list(FUTURE_TIMES))))
    options.append(("present", rng.choice(list(PRESENT_TIMES))))
    options.append(("habit", "EVERYDAY"))
    kind, gloss = rng.choices(options, weights=[5, 2, 2, 2, 1][: len(options)])[0]
    if kind == "none":
        return [], "", "default"
    english = {**PAST_TIMES, **FUTURE_TIMES, **PRESENT_TIMES, **HABIT_TIMES}[gloss]
    tense = {"past": "past", "future": "future", "present": "progressive", "habit": "habit"}[kind]
    return [gloss], english, tense


def _subject(rng, pronouns_only=False):
    key = rng.choice(PRONOUN_SUBJECTS if pronouns_only or rng.random() < 0.75 else list(SUBJECTS))
    return list(key), SUBJECTS[key]


# ── Sentence families ───────────────────────────────────────────────────────

def gen_verb_object(rng):
    verb = rng.choice(list(VERB_OBJECTS))
    obj = rng.choice(VERB_OBJECTS[verb])
    sg, (subj, _, poss, person) = _subject(rng)
    tg, tword, tense = _time(rng)
    negative = rng.random() < 0.2
    question = person == "2" and rng.random() < 0.35 and not negative
    if obj == "NAME":
        obj_en = f"{poss} name"
    elif obj in PERSONAL_THINGS and verb in {"LOSE", "FIND", "LOOK-FOR"}:
        obj_en = f"{poss} {THINGS[obj][1].split(' ', 1)[1]}"
    elif verb in {"EAT", "DRINK"} and obj == "MEDICINE":
        obj_en = "the medicine"
    else:
        obj_en = THINGS[obj][0]
    if tense == "default":
        tense = "present" if VERBS[verb][4] == "stative" else "progressive"
    if tense == "habit":
        tense = "present"
    rest = obj_en + (f" {tword}" if tword else "")
    english = _clause(subj, person, verb, rest, tense, negative, question)
    gloss = tg + sg + [verb] + [obj] + (["NOT"] if negative else []) + (["Q"] if question else [])
    if negative and rng.random() < 0.5:  # NOT before the verb is common too
        gloss = tg + sg + ["NOT", verb, obj]
    if obj in PERSONAL_THINGS and verb in {"LOSE", "FIND", "LOOK-FOR"} and subj in {"I", "you"} and rng.random() < 0.5:
        gloss.insert(gloss.index(obj), "MY" if subj == "I" else "YOUR")
    return gloss, _cap(english) + ("?" if question else ".")


def gen_motion(rng):
    verb = rng.choice(["GO", "COME"])
    place = rng.choice(list(PLACES))
    if verb == "COME" and place != "HOME":
        place_en = PLACES[place]
    else:
        place_en = PLACES[place]
    sg, (subj, _, _, person) = _subject(rng)
    tg, tword, tense = _time(rng)
    negative = rng.random() < 0.15
    question = person == "2" and rng.random() < 0.3 and not negative
    if tense in ("default",):
        tense = "progressive"
    if tense == "habit":
        tense = "present"
    rest = place_en + (f" {tword}" if tword else "")
    english = _clause(subj, person, verb, rest, tense, negative, question)
    gloss = tg + sg + [verb, place] + (["NOT"] if negative else []) + (["Q"] if question else [])
    return gloss, _cap(english) + ("?" if question else ".")


def gen_person_object(rng):
    verb = rng.choice(PERSON_VERBS)
    okey = rng.choice(list(PERSON_OBJECTS))
    sg, (subj, _, _, person) = _subject(rng)
    if list(okey) == sg or (okey == ("ME",) and sg == ["ME"]):
        return None
    tg, tword, tense = _time(rng)
    negative = rng.random() < 0.2
    question = person == "2" and rng.random() < 0.4 and not negative
    can = verb in {"HELP", "CALL"} and question and rng.random() < 0.6
    obj_en = PERSON_OBJECTS[okey]
    if verb == "WAIT":
        obj_en = "for " + obj_en
    if tense == "default":
        tense = "present" if VERBS[verb][4] == "stative" else "progressive"
    if tense == "habit":
        tense = "present"
    rest = obj_en + (f" {tword}" if tword else "")
    if can:
        english = f"can {subj} {VERBS[verb][0]} {rest}"
    else:
        english = _clause(subj, person, verb, rest, tense, negative, question)
    obj_gloss = list(PERSON_OBJECT_GLOSS.get(okey, okey))
    gloss = tg + sg + (["CAN"] if can else []) + [verb] + obj_gloss + (["NOT"] if negative else []) + (["Q"] if question else [])
    return gloss, _cap(english) + ("?" if question else ".")


def gen_adjective(rng):
    adj = rng.choice(list(ADJECTIVES))
    sg, (subj, _, _, person) = _subject(rng)
    tg, tword, tense = _time(rng)
    negative = rng.random() < 0.2
    question = person == "2" and rng.random() < 0.35 and not negative
    very = adj not in {"DEAF", "OK", "FINE", "LATE", "READY", "LOST"} and rng.random() < 0.25
    adj_en = ("very " if very else "") + ADJECTIVES[adj] + (f" {tword}" if tword and tense != "habit" else "")
    if tense == "past":
        be = BE_PAST[person]
    elif tense == "future":
        be = "will be"
    else:
        be = BE_PRESENT[person]
    if question:
        english = f"{be.split()[0]} {subj} {'be ' if tense == 'future' else ''}{adj_en}"
    elif negative:
        english = f"{subj} {'won\'t be' if tense == 'future' else be + ' not'} {adj_en}"
        english = english.replace("was not", "wasn't").replace("were not", "weren't")
        english = _contract(english)
    else:
        english = _contract(f"{subj} {be} {adj_en}")
    if tense == "habit":
        return None
    gloss = tg + sg + (["NOT"] if negative else []) + (["VERY"] if very else []) + [adj] + (["Q"] if question else [])
    return gloss, _cap(english) + ("?" if question else ".")


def gen_pain(rng):
    part = rng.choice(list(BODY))
    sg, (subj, _, poss, person) = _subject(rng, pronouns_only=True)
    question = person == "2" and rng.random() < 0.4
    tg, tword, tense = _time(rng, allow_future=False)
    if tense == "past":
        english = f"did {poss} {BODY[part]} hurt" if question else f"{poss} {BODY[part]} hurt {tword}"
    elif question:
        english = f"does {poss} {BODY[part]} hurt" + (f" {tword}" if tword and tense != "habit" else "")
    else:
        english = f"{poss} {BODY[part]} hurts" + (f" {tword}" if tword and tense != "habit" else "")
        if tense == "habit":
            english = f"{poss} {BODY[part]} hurts every day"
    order = rng.random()
    if order < 0.6:
        core = sg + [part, "HURT"]
    elif order < 0.85:
        core = [part, "HURT"] + sg
    else:
        core = sg + ["HURT", part]
    gloss = tg + core + (["Q"] if question else [])
    return gloss, _cap(english.strip()) + ("?" if question else ".")


def gen_where(rng):
    target = rng.choice(list(PLACE_NOUN))
    if rng.random() < 0.5:
        gloss = ["WHERE", target]
    else:
        gloss = [target, "WHERE"]
    return gloss, f"Where is {PLACE_NOUN[target]}?"


def gen_wh_verb(rng):
    kind = rng.random()
    sg, (subj, _, _, person) = _subject(rng, pronouns_only=True)
    tg, tword, tense = _time(rng)
    if tense in ("default", "habit"):
        tense = "progressive"
    suffix = f" {tword}" if tword else ""
    if kind < 0.35:
        verb = rng.choice(["GO", "LIVE", "WORK"])
        if verb == "GO":
            english = _clause(subj, person, "GO", suffix.strip(), tense, False, True)
        else:
            english = _clause(subj, person, verb, suffix.strip(), "present" if tense == "progressive" else tense, False, True)
        return tg + sg + [verb, "WHERE"], "Where " + english + "?"
    if kind < 0.65:
        verb = rng.choice(["WANT", "NEED", "LIKE", "EAT", "BUY", "LOOK-FOR"])
        t = tense if VERBS[verb][4] == "action" else ("present" if tense == "progressive" else tense)
        english = _clause(subj, person, verb, suffix.strip(), t, False, True)
        return tg + sg + [verb, "WHAT"], "What " + english + "?"
    if kind < 0.85:
        verb = rng.choice(["GO", "COME"])
        place = rng.choice(list(PLACES))
        # WHEN carries no time sign, so the gloss cannot mark past tense; ask about the future.
        english = _clause(subj, person, verb, PLACES[place], "future", False, True)
        return sg + [verb, place, "WHEN"], "When " + english + "?"
    verb = rng.choice(["GO", "COME", "WAIT", "SLEEP"])
    english = _clause(subj, person, verb, suffix.strip(), tense, False, True)
    return tg + sg + [verb, "WHY"], "Why " + english + "?"


def gen_name_place(rng):
    kind = rng.random()
    if kind < 0.4:
        name = _name(rng, FIRST_NAMES)
        return rng.choice([["ME", "NAME", _fs(name)], ["MY", "NAME", _fs(name)], ["NAME", "ME", _fs(name)]]), f"My name is {name}."
    if kind < 0.6:
        name = _name(rng, FIRST_NAMES)
        rel = rng.choice([("MOTHER", "mother"), ("FATHER", "father"), ("FRIEND", "friend"), ("CHILD", "child")])
        return ["MY", rel[0], "NAME", _fs(name)], f"My {rel[1]}'s name is {name}."
    if kind < 0.85:
        town = _name(rng, TOWNS)
        sg, (subj, _, _, person) = _subject(rng, pronouns_only=True)
        return sg + ["LIVE", _fs(town)], _cap(_clause(subj, person, "LIVE", f"in {town}", "present", False, False)) + "."
    town = _name(rng, TOWNS)
    return ["ME", "FROM", _fs(town)], f"I'm from {town}."


def gen_request(rng):
    verb = rng.choice(["HELP", "CALL", "WAIT", "WRITE", "REPEAT"])
    if verb == "HELP":
        obj = rng.choice([("ME",), ("MY", "MOTHER"), ("MY", "CHILD"), ("MY", "FRIEND")])
        obj_en = PERSON_OBJECTS[obj] if obj in PERSON_OBJECTS else "my child"
        if obj == ("MY", "CHILD"):
            obj_en = "my child"
    elif verb == "CALL":
        obj = rng.choice([("DOCTOR",), ("POLICE",), ("MY", "MOTHER"), ("MY", "FATHER"), ("TAXI",), ("AMBULANCE",)])
        obj_en = {("DOCTOR",): "a doctor", ("POLICE",): "the police", ("TAXI",): "a taxi",
                  ("AMBULANCE",): "an ambulance"}.get(obj) or PERSON_OBJECTS[obj]
    elif verb == "WAIT":
        obj, obj_en = ("ME",), "for me"
    elif verb == "WRITE":
        obj, obj_en = ("NAME",), "your name"
    else:
        obj, obj_en = (), "that"
    base = VERBS[verb][0] if verb in VERBS else "repeat"
    style = rng.random()
    if style < 0.45:
        gloss = ["PLEASE", verb] + list(obj) if rng.random() < 0.5 else [verb] + list(obj) + ["PLEASE"]
        english = f"Please {base} {obj_en}."
    elif style < 0.8:
        gloss = ["YOU", "CAN", verb] + list(obj) + ["Q"]
        english = f"Can you {base} {obj_en}?"
    else:
        gloss = ["YOU", verb] + list(obj) + ["PLEASE"]
        english = f"Can you {base} {obj_en}, please?"
    return gloss, english


def gen_number(rng):
    n = rng.choice([("TWO", "two"), ("THREE", "three"), ("FOUR", "four"), ("FIVE", "five")])
    thing = rng.choice(["CHILD", "BOOK", "TICKET", "BAG", "KEY", "PHONE", "FRIEND"])
    plural = {"CHILD": "children"}.get(thing, THINGS.get(thing, ("", "", thing.lower() + "s"))[2])
    sg, (subj, _, _, person) = _subject(rng, pronouns_only=True)
    verb = rng.choice(["HAVE", "NEED", "WANT"])
    english = _clause(subj, person, verb, f"{n[1]} {plural}", "present", False, False)
    return sg + [verb, n[0], thing], _cap(english) + "."


FAMILIES = [
    (gen_verb_object, 22), (gen_motion, 12), (gen_person_object, 12), (gen_adjective, 14),
    (gen_pain, 10), (gen_where, 5), (gen_wh_verb, 8), (gen_name_place, 7), (gen_request, 7),
    (gen_number, 3),
]

# Number glosses only appear in generated pairs; keep them out of the UI tray
# but known to the runtime.
EXTRA_GLOSSES = {"TWO", "THREE", "FOUR", "FIVE", "HAPPEN", "US", "THEM"}


def _one(rng):
    if rng.random() < 0.12:
        gloss, targets = rng.choice(FIXED)
        return list(gloss), targets[0]
    fns, weights = zip(*FAMILIES)
    for _ in range(10):
        out = rng.choices(fns, weights=weights)[0](rng)
        if out:
            return out
    return ["HELLO"], "Hello."


def _noise(rng, gloss):
    """Meaning-preserving recognizer noise: a repeated sign."""
    if len(gloss) > 1 and rng.random() < 0.08:
        i = rng.randrange(len(gloss))
        gloss = gloss[: i + 1] + [gloss[i]] + gloss[i + 1 :]
    return gloss


def generate(n, seed):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        gloss, english = _one(rng)
        if rng.random() < 0.15:  # two signed clauses -> two sentences
            gloss2, english2 = _one(rng)
            gloss, english = gloss + gloss2, english + " " + english2
        pairs.append({"src": PREFIX + " ".join(_noise(rng, gloss)), "tgt": english})
    return pairs


def check_vocabulary(pairs):
    known = set(sentence_vocab.GLOSSES) | EXTRA_GLOSSES
    unknown = {
        token
        for pair in pairs
        for token in pair["src"][len(PREFIX):].split()
        if token not in known and not token.startswith(sentence_vocab.FINGERSPELL_PREFIX)
    }
    if unknown:
        raise SystemExit(f"Generator used glosses missing from sentence_vocab: {sorted(unknown)}")


# ── ASLG-PC12 ───────────────────────────────────────────────────────────────

ASLG_PRONOUNS = {"X-I": "ME", "X-WE": "WE", "X-YOU": "YOU", "X-Y": "THEY", "X-HE": "HE", "X-SHE": "SHE",
                 "X-IT": "IT", "X-MY": "MY", "X-ITS": "ITS", "X-POSS": "POSS"}


def download_aslg(target):
    """Fetch ASLG-PC12 from Hugging Face (CC BY-NC 4.0); returns the local path or None."""
    try:
        from huggingface_hub import hf_hub_download

        root = target.parents[1]
        path = hf_hub_download("achrafothman/aslg_pc12", "data/train-00000-of-00001.parquet",
                               repo_type="dataset", local_dir=str(root))
        print(f"Downloaded ASLG-PC12 (CC BY-NC 4.0) to {path}")
        return Path(path)
    except Exception as error:
        print(f"Could not download ASLG-PC12 ({error}).")
        return None


def aslg_pairs(path, n, max_tokens, seed):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for row in rows:
        tokens = row["gloss"].split()
        if not tokens or len(tokens) > max_tokens:
            continue
        tokens = [ASLG_PRONOUNS.get(t, t[5:] if t.startswith("DESC-") else t) for t in tokens]
        out.append({"src": FORMAL_PREFIX + " ".join(tokens), "tgt": row["text"].strip()})
        if len(out) >= n:
            break
    return out


def main():
    parser = argparse.ArgumentParser(description="Build sentence-brain training pairs.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sentence_brain")
    parser.add_argument("--conversation", type=int, default=60000)
    parser.add_argument("--aslg", type=int, default=20000)
    parser.add_argument("--aslg-path", type=Path,
                        default=ROOT / "data/external_datasets/aslg_pc12/data/train-00000-of-00001.parquet")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    conv = generate(args.conversation, args.seed)
    check_vocabulary(conv)
    # Deduplicate by source so the same gloss never lands in both train and test.
    by_src = {}
    for pair in conv:
        by_src.setdefault(pair["src"], pair)
    conv = list(by_src.values())
    rng = random.Random(args.seed)
    rng.shuffle(conv)
    n_test = min(1500, len(conv) // 20)
    splits = {"test": conv[:n_test], "val": conv[n_test: 2 * n_test], "train": conv[2 * n_test:]}

    if args.aslg and not args.aslg_path.is_file():
        args.aslg_path = download_aslg(args.aslg_path)
    if args.aslg and args.aslg_path and args.aslg_path.is_file():
        formal = aslg_pairs(args.aslg_path, args.aslg + 1000, 14, args.seed)
        splits["val_formal"] = formal[:1000]
        splits["train"] += formal[1000:]
        rng.shuffle(splits["train"])
    elif args.aslg:
        print(f"ASLG-PC12 not found at {args.aslg_path}; training on conversation pairs only.")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} pairs")


if __name__ == "__main__":
    main()
