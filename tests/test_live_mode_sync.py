import unittest

import app


class TestLiveModeSync(unittest.TestCase):
    """The server session must follow the mode the page shows."""

    def setUp(self):
        self.client = app.app.test_client()
        self.username = "mode_sync_tester"
        with self.client.session_transaction() as session:
            session["username"] = self.username

    def _server_mode(self):
        return app._get_inference_session(self.username)["mode"]

    def test_camera_start_restores_letter_mode_after_word_mode(self):
        self.client.post("/api/mode", json={"mode": "word"})
        self.assertEqual(self._server_mode(), "word")

        response = self.client.post("/api/camera/start", json={"mode": "letter"})

        self.assertEqual(response.get_json()["mode"], "letter")
        self.assertEqual(self._server_mode(), "letter")

    def test_camera_start_without_body_defaults_to_letter(self):
        self.client.post("/api/mode", json={"mode": "word"})
        self.client.post("/api/camera/start")
        self.assertEqual(self._server_mode(), "letter")

    def test_landmark_requests_carry_the_page_mode(self):
        self.client.post("/api/camera/start", json={"mode": "letter"})
        data = self.client.post(
            "/api/infer_landmarks", json={"landmarks": [], "mode": "word"}
        ).get_json()
        self.assertEqual(data["mode"], "word")

        data = self.client.post("/api/infer_landmarks", json={"landmarks": []}).get_json()
        self.assertEqual(data["mode"], "word", "a request without a mode keeps the current one")

    def test_invalid_mode_is_ignored(self):
        self.client.post("/api/camera/start", json={"mode": "letter"})
        self.client.post("/api/infer_landmarks", json={"landmarks": [], "mode": "sentence"})
        self.assertEqual(self._server_mode(), "letter")


if __name__ == "__main__":
    unittest.main()
