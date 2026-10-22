import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import select

import app
import database
from learning_mode import ensure_additional_alphabet_lessons, record_practice, seed_starter_course, sign_mastery, sync_avatar_catalog


class TestLearningEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "empty"
        self.database_path = Path(self.temp_dir.name) / "learning.sqlite3"
        database.initialize_database(f"sqlite:///{self.database_path.as_posix()}", str(self.data_dir))
        seed_starter_course()
        sync_avatar_catalog()
        ensure_additional_alphabet_lessons()
        database.create_user("learner", "test-hash")
        database.create_user("other-learner", "other-hash")
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = "learner"
        courses = self.client.get("/api/learning/courses").get_json()["courses"]
        self.course = courses[0]
        self.lessons = [lesson for module in self.course["modules"] for lesson in module["lessons"]]
        self.first_lesson_id = self.lessons[0]["id"]

    def tearDown(self):
        if database._engine is not None:
            database._engine.dispose()
        self.temp_dir.cleanup()

    def advance_to_test(self, lesson_id):
        for section in ("demonstration", "explanation", "practice", "exercise"):
            response = self.client.post(
                f"/api/learning/lessons/{lesson_id}/progress",
                json={"section": section},
            )
            self.assertEqual(response.status_code, 200, response.get_json())
        exercise = self.client.post(
            f"/api/learning/lessons/{lesson_id}/exercise",
            json={"answer_index": 0},
        )
        self.assertEqual(exercise.status_code, 200, exercise.get_json())
        response = self.client.post(
            f"/api/learning/lessons/{lesson_id}/progress",
            json={"section": "test"},
        )
        self.assertEqual(response.status_code, 200, response.get_json())

    def answer_quiz(self, quiz_data, get_correct, count=None):
        with database._session() as session:
            stored_session = session.get(database.QuizSession, quiz_data["session_id"])
            answer_keys = {
                question["id"]: question["answer_index"]
                for question in json.loads(stored_session.questions_json)
            }
        for question in quiz_data["questions"][:count]:
            options = question["options"]
            answer_index = answer_keys[question["id"]]
            option_index = answer_index if get_correct else (answer_index + 1) % len(options)
            response = self.client.post(
                f"/api/learning/quiz/{quiz_data['session_id']}/answer",
                json={"question_id": question["id"], "option_index": option_index},
            )
            self.assertEqual(response.status_code, 200, response.get_json())

    def test_course_lesson_ordering_and_locks(self):
        self.assertEqual(self.course["total_lessons"], 9)
        self.assertEqual(self.lessons[0]["status"], "unlocked")
        self.assertEqual(self.lessons[1]["status"], "locked")
        self.assertEqual(self.client.get(f"/api/learning/lessons/{self.lessons[1]['id']}").status_code, 403)
        self.assertEqual(self.client.get("/api/learning/courses/999999").status_code, 404)

    def test_server_requires_order_and_mini_exercise_before_quiz(self):
        lesson_id = self.first_lesson_id
        skipped = self.client.post(f"/api/learning/lessons/{lesson_id}/progress", json={"section": "test"})
        self.assertEqual(skipped.status_code, 409)
        early_quiz = self.client.post(f"/api/learning/lessons/{lesson_id}/quiz/start")
        self.assertEqual(early_quiz.status_code, 403)
        self.advance_to_test(lesson_id)
        quiz = self.client.post(f"/api/learning/lessons/{lesson_id}/quiz/start")
        self.assertEqual(quiz.status_code, 200)
        self.assertGreaterEqual(len(quiz.get_json()["questions"]), 4)

    def test_answer_keys_are_private_and_a_pass_unlocks_next_lesson(self):
        self.advance_to_test(self.first_lesson_id)
        quiz = self.client.post(f"/api/learning/lessons/{self.first_lesson_id}/quiz/start").get_json()
        avatar_question = next(question for question in quiz["questions"] if question["type"] == "avatar_identification")
        self.assertTrue(avatar_question["avatar_available"])
        self.assertNotIn("avatar_asset", avatar_question)
        avatar_demo = self.client.get(
            f"/api/learning/quiz/{quiz['session_id']}/question/{avatar_question['id']}/demo"
        ).get_json()
        self.assertEqual(avatar_demo["asset"]["path"], "/static/avatar/sigml/asl/HELLO.sigml")
        for question in quiz["questions"]:
            self.assertNotIn("answer_index", question)
            self.assertNotIn("avatar_label", question)
            self.assertNotIn("recognition_label", question)
        self.answer_quiz(quiz, get_correct=True)
        result = self.client.post(f"/api/learning/quiz/{quiz['session_id']}/complete")
        self.assertEqual(result.status_code, 200, result.get_json())
        self.assertTrue(result.get_json()["passed"])
        course = self.client.get(f"/api/learning/courses/{self.course['id']}").get_json()
        updated = [lesson for module in course["modules"] for lesson in module["lessons"]]
        self.assertEqual(updated[0]["status"], "completed")
        self.assertEqual(updated[1]["status"], "unlocked")
        self.assertTrue(self.client.get("/api/learning/mastery").get_json()["signs"])

    def test_failed_attempt_is_saved_and_can_be_retried(self):
        self.advance_to_test(self.first_lesson_id)
        quiz = self.client.post(f"/api/learning/lessons/{self.first_lesson_id}/quiz/start").get_json()
        self.answer_quiz(quiz, get_correct=False)
        result = self.client.post(f"/api/learning/quiz/{quiz['session_id']}/complete").get_json()
        self.assertFalse(result["passed"])
        self.assertEqual(result["percentage"], 0)
        retry = self.client.post(f"/api/learning/lessons/{self.first_lesson_id}/quiz/start")
        self.assertEqual(retry.status_code, 200, retry.get_json())
        self.assertEqual(retry.get_json()["attempt_number"], 2)
        self.assertEqual(self.client.post(f"/api/learning/quiz/{quiz['session_id']}/complete").status_code, 409)

    def test_quiz_resumes_and_isolated_to_owner(self):
        self.advance_to_test(self.first_lesson_id)
        quiz = self.client.post(f"/api/learning/lessons/{self.first_lesson_id}/quiz/start").get_json()
        question = quiz["questions"][0]
        self.client.post(f"/api/learning/quiz/{quiz['session_id']}/answer", json={"question_id": question["id"], "option_index": 0})
        resumed = self.client.get(f"/api/learning/quiz/{quiz['session_id']}/resume")
        self.assertEqual(resumed.status_code, 200)
        self.assertEqual(resumed.get_json()["answered_ids"], [question["id"]])
        lesson_state = self.client.get(f"/api/learning/lessons/{self.first_lesson_id}").get_json()
        self.assertEqual(lesson_state["active_quiz_session"], quiz["session_id"])
        self.assertEqual(lesson_state["current_section"], "test")
        self.assertEqual(self.client.get(f"/api/learning/quiz/{quiz['session_id']}/question/{question['id']}/demo").status_code, 200)
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = "other-learner"
        self.assertEqual(self.client.get(f"/api/learning/quiz/{quiz['session_id']}/resume").status_code, 404)
        self.assertEqual(self.client.get("/api/learning/progress").get_json()["recent_tests"], [])
        self.assertEqual(self.client.get(f"/api/learning/quiz/{quiz['session_id']}/question/{question['id']}/demo").status_code, 404)
        self.assertEqual(self.client.post(f"/api/learning/quiz/{quiz['session_id']}/answer", json={"question_id": question["id"], "option_index": 0}).status_code, 404)

    def test_practice_result_and_mastery_are_persisted(self):
        with database._session() as session, session.begin():
            sign = session.scalar(select(database.Sign).where(database.Sign.name == "HELLO"))
            sign.recognition_label = "A"
            sign_id = sign.id
        result = record_practice("learner", self.first_lesson_id, sign_id, "A", 0.93)
        self.assertTrue(result["correct"])
        mastery = sign_mastery("learner")
        self.assertEqual(mastery[0]["score"], 93.0)
        with database._session() as session:
            saved = session.scalars(select(database.PracticeAttempt)).all()
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].predicted_label, "A")

    def test_camera_quiz_uses_existing_prediction_and_records_attempt(self):
        alphabet_lesson_id = self.lessons[4]["id"]
        with database._session() as session, session.begin():
            for lesson_summary in self.lessons[:4]:
                progress = database.UserLessonProgress(
                    username="learner",
                    lesson_id=lesson_summary["id"],
                    status="completed",
                    progress_percentage=100,
                    exercise_completed=True,
                )
                session.add(progress)
            current = database.UserLessonProgress(
                username="learner",
                lesson_id=alphabet_lesson_id,
                status="in_progress",
                current_section="test",
                progress_percentage=80,
                exercise_completed=True,
            )
            session.add(current)
            letter_a = session.scalar(select(database.Sign).where(database.Sign.name == "A"))
            letter_a.recognition_label = "A"

        quiz = self.client.post(f"/api/learning/lessons/{alphabet_lesson_id}/quiz/start").get_json()
        recognition_question = next(question for question in quiz["questions"] if question["type"] == "sign_recognition")
        rejected_choice = self.client.post(
            f"/api/learning/quiz/{quiz['session_id']}/answer",
            json={"question_id": recognition_question["id"], "option_index": 0},
        )
        self.assertEqual(rejected_choice.status_code, 422)

        fake_frame = SimpleNamespace(shape=(480, 640, 3))
        fake_landmarks = [SimpleNamespace(x=0.1, y=0.1) for _ in range(21)]
        fake_detector = SimpleNamespace(
            find_hands=lambda frame, draw=False: (frame, SimpleNamespace(hand_landmarks=[fake_landmarks])),
            get_landmarks=lambda frame, hand_no=0: [[index, 0.1, 0.1] for index in range(21)],
        )
        with patch.object(app, "_decode_posted_frame", return_value=(fake_frame, "")), \
             patch.object(app, "_load_inference_modules"), \
             patch.object(app, "_inference_status", {"ready": True, "error": ""}), \
             patch.object(app, "_detector", fake_detector), \
             patch.object(app, "_extractor", SimpleNamespace(extract_features=lambda landmarks: [0.0] * 42)), \
             patch.object(app, "_is_plausible_hand", return_value=True), \
             patch.object(app, "_predict_static_sign", return_value=("A", 0.96)):
            response = self.client.post(
                f"/api/learning/quiz/{quiz['session_id']}/question/{recognition_question['id']}/recognition",
                data={},
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(response.get_json()["correct"])
        with database._session() as session:
            attempt = session.scalar(select(database.PracticeAttempt))
            self.assertEqual(attempt.target_label, "A")
            self.assertEqual(attempt.predicted_label, "A")

    def test_authenticated_full_learning_cycle_survives_relogin(self):
        client = app.app.test_client()
        registration = client.post("/register", json={"username": "course-user", "password": "course-password"})
        self.assertEqual(registration.status_code, 200, registration.get_json())
        login = client.post("/login", json={"username": "course-user", "password": "course-password"})
        self.assertEqual(login.status_code, 200, login.get_json())
        course = client.get("/api/learning/courses").get_json()["courses"][0]
        lesson_id = course["modules"][0]["lessons"][0]["id"]
        self.assertEqual(client.get(f"/api/learning/lessons/{lesson_id}/content").status_code, 200)

        for section in ("demonstration", "explanation", "practice", "exercise"):
            self.assertEqual(client.post(f"/api/learning/lessons/{lesson_id}/progress", json={"section": section}).status_code, 200)
        self.assertEqual(client.post(f"/api/learning/lessons/{lesson_id}/exercise", json={"answer_index": 0}).status_code, 200)
        self.assertEqual(client.post(f"/api/learning/lessons/{lesson_id}/progress", json={"section": "test"}).status_code, 200)
        quiz = client.post(f"/api/learning/lessons/{lesson_id}/quiz/start").get_json()
        self.assertEqual(len(quiz["questions"]), 5)
        for question in quiz["questions"]:
            with database._session() as session:
                persisted = session.get(database.QuizSession, quiz["session_id"])
                answer_index = next(item["answer_index"] for item in json.loads(persisted.questions_json) if item["id"] == question["id"])
            response = client.post(
                f"/api/learning/quiz/{quiz['session_id']}/answer",
                json={"question_id": question["id"], "option_index": answer_index},
            )
            self.assertEqual(response.status_code, 200, response.get_json())
        result = client.post(f"/api/learning/quiz/{quiz['session_id']}/complete").get_json()
        self.assertTrue(result["passed"])

        client.get("/logout")
        self.assertEqual(client.post("/login", json={"username": "course-user", "password": "course-password"}).status_code, 200)
        progress = client.get(f"/api/learning/courses/{course['id']}").get_json()
        lesson_statuses = [lesson["status"] for module in progress["modules"] for lesson in module["lessons"]]
        self.assertEqual(lesson_statuses[:2], ["completed", "unlocked"])
        self.assertEqual(len(client.get("/api/learning/progress").get_json()["recent_tests"]), 1)


if __name__ == "__main__":
    unittest.main()