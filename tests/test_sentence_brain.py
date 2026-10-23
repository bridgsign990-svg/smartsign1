import unittest
from unittest.mock import patch

import app
import sentence_vocab
from sentence_brain import SentenceBrain
from word_assembler import WordAssembler


class TestSentenceVocab(unittest.TestCase):
    def test_recognizer_labels_map_to_glosses(self):
        self.assertEqual(sentence_vocab.normalize_gloss("THANK_YOU"), "THANK-YOU")
        self.assertEqual(sentence_vocab.normalize_gloss("hello"), "HELLO")
        self.assertEqual(sentence_vocab.normalize_gloss("how much"), "HOW-MUCH")

    def test_unknown_words_are_finger_spelled(self):
        self.assertEqual(sentence_vocab.normalize_gloss("John"), "fs-JOHN")
        self.assertEqual(sentence_vocab.normalize_gloss("fs-lilongwe"), "fs-LILONGWE")
        self.assertEqual(sentence_vocab.normalize_gloss("  "), "")


class _NoModelBrain(SentenceBrain):
    def _load(self):
        self.last_error = "test: no model"


class _FakeModelBrain(SentenceBrain):
    """Model stand-in: echoes the gloss sequence so ranking logic can be tested."""

    def _load(self):
        self.encoder = self.decoder = self.tokenizer = object()

    def _generate(self, glosses):
        if glosses == ["ME", "HEAD", "HURT"]:
            return [("My head hurts.", -0.1), ("I have a headache.", -0.9)]
        if glosses == ["ME", "HEAD", "HELP"]:
            return [("Help my head.", -0.2)]
        if glosses == ["ME", "NAME", "fs-JOHN"]:
            return [("My name is Jon.", -0.1), ("My name is John.", -0.3)]
        return [(" ".join(glosses), -1.0)]


class TestSentenceBrain(unittest.TestCase):
    def test_literal_reading_when_model_missing(self):
        brain = _NoModelBrain()
        self.assertFalse(brain.is_available())
        [only] = brain.suggest(["ME", "NAME", "fs-JOHN"])
        self.assertEqual(only["text"], "Me name John.")
        self.assertEqual(only["source"], "literal")
        [question] = brain.suggest(["YOU", "UNDERSTAND", "Q"])
        self.assertEqual(question["text"], "You understand?")

    def test_empty_input_gives_no_suggestions(self):
        self.assertEqual(_FakeModelBrain().suggest([]), [])
        self.assertEqual(_FakeModelBrain().suggest(["", None]), [])

    def test_candidates_that_lose_a_spelled_name_are_dropped(self):
        # fs-BUKAMA reaches the model as the stand-in John; "Jon" loses the name.
        texts = [c["text"] for c in _FakeModelBrain().suggest(["ME", "NAME", "fs-BUKAMA"])]
        self.assertEqual(texts, ["My name is Bukama."])

    def test_uncertain_sign_tries_its_alternative(self):
        signs = ["ME", "HEAD", {"gloss": "HELP", "confidence": 0.4,
                                "alternatives": [{"gloss": "HURT", "confidence": 0.35}]}]
        suggestions = _FakeModelBrain().suggest(signs)
        texts = [c["text"] for c in suggestions]
        self.assertIn("My head hurts.", texts)
        self.assertIn("Help my head.", texts)
        self.assertAlmostEqual(sum(c["score"] for c in suggestions), 1.0, places=2)

    def test_confident_sign_is_not_swapped(self):
        signs = ["ME", "HEAD", {"gloss": "HURT", "confidence": 0.9,
                                "alternatives": [{"gloss": "HELP", "confidence": 0.05}]}]
        texts = [c["text"] for c in _FakeModelBrain().suggest(signs)]
        self.assertEqual(texts, ["My head hurts.", "I have a headache."])


    def test_spelled_names_use_stand_ins_and_are_restored(self):
        glosses, mapping = SentenceBrain._stand_in_names(["ME", "NAME", "fs-BUKAMA", "ME", "FROM", "fs-ZOMBA"])
        self.assertEqual(glosses, ["ME", "NAME", "fs-JOHN", "ME", "FROM", "fs-MARY"])
        self.assertEqual(
            SentenceBrain._restore_names("My name is John. I'm from Mary.", mapping),
            "My name is Bukama. I'm from Zomba.",
        )
        # A real name equal to a stand-in is never swapped for another person.
        glosses, mapping = SentenceBrain._stand_in_names(["ME", "NAME", "fs-JOHN"])
        self.assertEqual(SentenceBrain._restore_names("My name is " + glosses[2][3:].capitalize() + ".", mapping),
                         "My name is John.")

    def test_garbled_or_messy_output_is_cleaned(self):
        self.assertEqual(SentenceBrain._clean("I don�t understand."), "")
        self.assertEqual(SentenceBrain._clean("My name is Chikondi.."), "My name is Chikondi.")
        self.assertEqual(SentenceBrain._clean("Wait ..."), "Wait...")
        self.assertEqual(SentenceBrain._clean("Hello ."), "Hello.")

    def test_unlikely_alternatives_are_hidden(self):
        class Confident(_FakeModelBrain):
            def _generate(self, glosses):
                return [("I need help.", -0.05), ("I needs help.", -3.0)]

        texts = [c["text"] for c in Confident().suggest(["ME", "NEED", "HELP"])]
        self.assertEqual(texts, ["I need help."])


class TestRealSentenceModel(unittest.TestCase):
    """Runs the exported model when it is installed (Git LFS checkout)."""

    @classmethod
    def setUpClass(cls):
        cls.brain = SentenceBrain()
        if not cls.brain.is_available():
            raise unittest.SkipTest(f"sentence model not installed: {cls.brain.last_error}")

    def top(self, glosses):
        return self.brain.suggest(glosses.split())[0]["text"]

    def test_everyday_sentences(self):
        self.assertEqual(self.top("ME HEAD HURT"), "My head hurts.")
        self.assertEqual(self.top("YESTERDAY ME GO HOSPITAL"), "I went to the hospital yesterday.")
        self.assertEqual(self.top("YOU NAME WHAT"), "What is your name?")
        self.assertEqual(self.top("YOU UNDERSTAND Q"), "Do you understand?")

    def test_rare_spelled_name_is_copied_exactly(self):
        self.assertEqual(self.top("MY FATHER NAME fs-BUKAMA"), "My father's name is Bukama.")


class TestAssemblerGlosses(unittest.TestCase):
    def test_gesture_words_and_spelled_words_become_glosses(self):
        assembler = WordAssembler()
        assembler.push_word("HELLO")
        for letter in "JOHN":
            assembler.push_letter(letter)
        assembler.push_word("THANK_YOU")
        self.assertEqual(assembler.glosses, ["HELLO", "fs-JOHN", "THANK-YOU"])
        self.assertEqual(assembler.tick(hand_present=True)["glosses"], ["HELLO", "fs-JOHN", "THANK-YOU"])
        assembler.reset()
        self.assertEqual(assembler.glosses, [])


class TestSentenceEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        with self.client.session_transaction() as session:
            session["username"] = "sentence_tester"

    def test_vocab_lists_groups(self):
        data = self.client.get("/api/sentence/vocab").get_json()
        names = [group["name"] for group in data["groups"]]
        self.assertEqual(names, list(sentence_vocab.GROUPS))
        people = next(group for group in data["groups"] if group["name"] == "People")
        self.assertIn("ME", people["glosses"])
        self.assertIn("brain_available", data)

    def test_suggest_validates_input(self):
        self.assertEqual(self.client.post("/api/sentence/suggest", json={"signs": "ME"}).status_code, 400)
        self.assertEqual(self.client.post("/api/sentence/suggest", json={"signs": ["ME"] * 41}).status_code, 400)

    def test_suggest_returns_ranked_sentences(self):
        with patch("sentence_brain._brain", _FakeModelBrain()):
            data = self.client.post("/api/sentence/suggest", json={"signs": ["ME", "HEAD", "HURT"]}).get_json()
        self.assertEqual(data["suggestions"][0]["text"], "My head hurts.")

    def test_clear_empties_signed_glosses(self):
        self.client.post("/api/camera/start", json={"mode": "word"})
        state = app._get_inference_session("sentence_tester")
        state["assembler"].push_word("HELLO")
        self.assertEqual(state["assembler"].glosses, ["HELLO"])
        self.client.post("/api/sentence/clear")
        self.assertEqual(state["assembler"].glosses, [])


if __name__ == "__main__":
    unittest.main()
