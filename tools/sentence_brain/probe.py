"""
Print the sentence brain's suggestions for hand-written sign sequences.

The sequences mix patterns the generator covers with unusual orders and
combinations it never produced, so this shows how the exported model behaves
beyond its training templates. Add the sequences your testers actually sign.

    python tools/sentence_brain/probe.py
    python tools/sentence_brain/probe.py "ME WANT GO HOME" "YOU HUNGRY Q"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentence_brain import SentenceBrain  # noqa: E402

PROBES = [
    "HELLO", "ME NEED HELP", "ME HEAD HURT", "HEAD HURT ME", "ME FEEL NOT GOOD",
    "ME NOT FEEL GOOD HEAD HURT", "YOU NAME WHAT", "ME NAME fs-CHIKONDI", "WHERE TOILET",
    "ME WANT WATER PLEASE", "YESTERDAY ME GO HOSPITAL", "TOMORROW ME GO SCHOOL",
    "YOU UNDERSTAND Q", "ME NOT UNDERSTAND", "PLEASE REPEAT", "ME DEAF PLEASE WRITE",
    "CALL AMBULANCE", "MY CHILD SICK", "MY BABY HUNGRY", "ME LOSE MY PHONE",
    "HOW-MUCH", "THANK-YOU", "ME LIVE fs-ZOMBA", "YOU GO WHERE", "ME HUNGRY ME WANT FOOD",
    # beyond the generator's templates
    "ME WANT GO HOME", "ME TIRED WANT SLEEP", "STOMACH HURT VERY", "YOU HELP ME FIND TOILET",
    "DOCTOR WHERE", "ME SCARED", "FIRE HELP",
]


def main():
    probes = sys.argv[1:] or PROBES
    brain = SentenceBrain()
    if not brain.is_available():
        print(f"Model unavailable: {brain.last_error}")
    for probe in probes:
        started = time.perf_counter()
        suggestions = brain.suggest(probe.split())
        ms = (time.perf_counter() - started) * 1000
        shown = " | ".join(f"{s['text']} ({s['score']:.0%})" for s in suggestions)
        print(f"{probe:<30} -> {shown}  [{ms:.0f} ms]")


if __name__ == "__main__":
    main()
