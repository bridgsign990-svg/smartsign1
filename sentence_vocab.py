"""
BridgeSign – Sentence vocabulary
================================
The everyday sign glosses the sentence brain understands, grouped for the
Sentence-mode sign tray. Shared by the training-pair generator
(tools/sentence_brain/build_pairs.py), the runtime brain (sentence_brain.py)
and the browser UI (/api/sentence/vocab).

Gloss conventions
-----------------
- Upper case, multi-word signs joined with "-" (THANK-YOU, HOW-MUCH).
- ME/YOU/HE/SHE/WE/THEY for pronouns; MY/YOUR for possessives.
- A trailing "Q" marks a yes/no question (eyebrows up in signing).
- Finger-spelled words that are not glosses are passed as "fs-WORD".
"""

# Recognizer labels (gesture model) that differ from brain glosses.
RECOGNIZER_TO_GLOSS = {
    "THANK_YOU": "THANK-YOU",
}

GROUPS = {
    "Social": [
        "HELLO", "GOODBYE", "THANK-YOU", "PLEASE", "SORRY", "YES", "NO", "OK",
        "NICE", "MEET", "WELCOME", "EXCUSE-ME", "GOOD", "MORNING", "AFTERNOON", "NIGHT",
    ],
    "People": [
        "ME", "YOU", "HE", "SHE", "WE", "THEY", "MY", "YOUR",
        "MOTHER", "FATHER", "FRIEND", "CHILD", "BABY", "DOCTOR", "TEACHER", "POLICE", "INTERPRETER",
    ],
    "Actions": [
        "NEED", "WANT", "LIKE", "HAVE", "GO", "COME", "EAT", "DRINK", "SEE", "KNOW",
        "UNDERSTAND", "HELP", "CALL", "LOOK-FOR", "LIVE", "WORK", "LEARN", "WAIT",
        "SLEEP", "BUY", "PAY", "FIND", "LOSE", "FEEL", "HURT", "WRITE", "REPEAT", "STOP",
    ],
    "Things": [
        "WATER", "FOOD", "MEDICINE", "MONEY", "PHONE", "BAG", "KEY", "BOOK", "TICKET",
        "TAXI", "CAR", "BREAD", "TEA", "AMBULANCE", "NAME", "TIME",
    ],
    "Places": [
        "HOME", "SCHOOL", "HOSPITAL", "CLINIC", "CHURCH", "MARKET", "SHOP", "BANK",
        "TOILET", "TOWN", "PHARMACY", "POLICE-STATION",
    ],
    "Body": ["HEAD", "STOMACH", "CHEST", "BACK", "LEG", "ARM", "EYE", "EAR", "TOOTH", "THROAT", "HAND", "FOOT"],
    "Feelings": [
        "TIRED", "HUNGRY", "THIRSTY", "SICK", "HAPPY", "SAD", "COLD", "HOT", "SCARED",
        "ANGRY", "FINE", "BUSY", "LOST", "LATE", "READY", "WORRIED", "DIZZY", "DEAF", "BAD",
    ],
    "Questions": ["WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW", "HOW-MUCH", "Q"],
    "Time": ["NOW", "TODAY", "TOMORROW", "YESTERDAY", "LATER", "EVERYDAY"],
    "Other": ["NOT", "VERY", "AGAIN", "SLOW", "CAN", "EMERGENCY", "FIRE", "DANGER", "FOR", "FROM"],
}

GLOSSES = frozenset(gloss for group in GROUPS.values() for gloss in group)
FINGERSPELL_PREFIX = "fs-"


def normalize_gloss(token):
    """Map a recognizer label or typed word onto a brain gloss (or fs-WORD)."""
    raw = str(token or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith(FINGERSPELL_PREFIX):
        return FINGERSPELL_PREFIX + raw[len(FINGERSPELL_PREFIX):].upper()
    upper = raw.upper().replace(" ", "-")
    upper = RECOGNIZER_TO_GLOSS.get(upper, upper).replace("_", "-")
    if upper in GLOSSES:
        return upper
    return FINGERSPELL_PREFIX + upper
