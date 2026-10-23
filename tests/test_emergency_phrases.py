import tempfile
import unittest
from pathlib import Path

import app
import database


class TestEmergencyPhraseEndpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database.initialize_database(
            f"sqlite:///{(Path(self.temp_dir.name) / 'emergency.sqlite3').as_posix()}",
            str(Path(self.temp_dir.name) / "data"),
        )
        self.username = "emergency-test-user"
        database.create_user(self.username, "test-hash")
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = self.username

    def tearDown(self):
        if database._engine is not None:
            database._engine.dispose()
        self.temp_dir.cleanup()

    def test_custom_phrase_requires_nonempty_text_and_enforces_limit(self):
        empty = self.client.post("/api/emergency/custom", json={"phrase": "  "})
        too_long = self.client.post("/api/emergency/custom", json={"phrase": "x" * 501})
        malformed = self.client.post("/api/emergency/custom", json={"phrase": ["help"]})

        self.assertEqual(empty.status_code, 400)
        self.assertEqual(too_long.status_code, 400)
        self.assertEqual(malformed.status_code, 400)

    def test_custom_phrase_can_be_added_and_deleted(self):
        added = self.client.post("/api/emergency/custom", json={"phrase": "Please help me"})
        self.assertEqual(added.status_code, 200)
        phrase_id = added.get_json()["id"]
        self.assertTrue(phrase_id.startswith("c_"))

        listing = self.client.get("/api/emergency/all")
        self.assertEqual(listing.get_json()[phrase_id], "Please help me")

        deleted = self.client.delete(f"/api/emergency/custom/{phrase_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertNotIn(phrase_id, self.client.get("/api/emergency/all").get_json())


if __name__ == "__main__":
    unittest.main()
