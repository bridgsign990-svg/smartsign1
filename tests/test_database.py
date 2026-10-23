import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect

import database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "legacy"
        self.data_dir.mkdir()
        self.database_path = Path(self.temp_dir.name) / "bridgesign.sqlite3"
        (self.data_dir / "users.json").write_text(
            json.dumps({"legacy-user": {"password": "hashed-password"}}),
            encoding="utf-8",
        )
        (self.data_dir / "phrases_legacy-user.json").write_text(
            json.dumps({"custom-1": "Please call for help"}),
            encoding="utf-8",
        )
        database.initialize_database(
            f"sqlite:///{self.database_path.as_posix()}",
            str(self.data_dir),
        )

    def tearDown(self):
        if database._engine is not None:
            database._engine.dispose()
        self.temp_dir.cleanup()

    def test_imports_existing_accounts_and_phrases(self):
        self.assertEqual(database.get_user("legacy-user"), {"password": "hashed-password"})
        self.assertEqual(
            database.load_user_phrases("legacy-user"),
            {"custom-1": "Please call for help"},
        )

    def test_account_and_phrase_changes_persist(self):
        self.assertTrue(database.create_user("new-user", "new-hash"))
        self.assertFalse(database.create_user("new-user", "another-hash"))
        database.save_user_phrases("new-user", {"custom-2": "I need assistance"})

        database.initialize_database(
            f"sqlite:///{self.database_path.as_posix()}",
            str(Path(self.temp_dir.name) / "empty"),
        )

        self.assertEqual(database.get_user("new-user"), {"password": "new-hash"})
        self.assertEqual(
            database.load_user_phrases("new-user"),
            {"custom-2": "I need assistance"},
        )

    def test_learning_tables_are_initialized(self):
        table_names = set(inspect(database._engine).get_table_names())
        self.assertTrue(
            {
                "courses",
                "modules",
                "lessons",
                "signs",
                "lesson_signs",
                "user_lesson_progress",
                "quiz_attempts",
                "quiz_questions",
                "quiz_sessions",
                "practice_attempts",
                "user_sign_mastery",
            }.issubset(table_names)
        )

    def test_upgrades_phase_one_progress_table(self):
        legacy_path = Path(self.temp_dir.name) / "phase-one.sqlite3"
        legacy_engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
        with legacy_engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE user_lesson_progress ("
                "id INTEGER PRIMARY KEY, username VARCHAR(80) NOT NULL, lesson_id INTEGER NOT NULL, "
                "status VARCHAR(30) NOT NULL, progress_percentage INTEGER NOT NULL, best_score FLOAT NOT NULL, "
                "attempts INTEGER NOT NULL, completed_at DATETIME)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE signs ("
                "id INTEGER PRIMARY KEY, name VARCHAR(120) NOT NULL, meaning TEXT NOT NULL, "
                "description TEXT NOT NULL, usage TEXT NOT NULL, avatar_animation VARCHAR(255), "
                "category VARCHAR(80) NOT NULL, difficulty VARCHAR(40) NOT NULL, recognition_label VARCHAR(80), "
                "video_url VARCHAR(500), image_url VARCHAR(500), common_mistakes_json TEXT NOT NULL)"
            )
            connection.exec_driver_sql(
                "INSERT INTO signs (id, name, meaning, description, usage, category, difficulty, common_mistakes_json) "
                "VALUES (1, 'HELLO', 'Greeting', 'Existing description', 'Existing usage', 'greetings', 'beginner', '[]')"
            )
            connection.exec_driver_sql(
                "INSERT INTO user_lesson_progress "
                "(id, username, lesson_id, status, progress_percentage, best_score, attempts, completed_at) "
                "VALUES (7, 'legacy-learner', 12, 'completed', 100, 92.5, 2, '2025-01-01 00:00:00')"
            )
            connection.exec_driver_sql(
                "CREATE TABLE quiz_attempts ("
                "id INTEGER PRIMARY KEY, username VARCHAR(80) NOT NULL, lesson_id INTEGER NOT NULL, "
                "score INTEGER NOT NULL, total_questions INTEGER NOT NULL, percentage FLOAT NOT NULL, "
                "passed BOOLEAN NOT NULL, completed_at DATETIME)"
            )
            connection.exec_driver_sql(
                "INSERT INTO quiz_attempts "
                "(id, username, lesson_id, score, total_questions, percentage, passed, completed_at) "
                "VALUES (5, 'legacy-learner', 12, 9, 10, 90, 1, '2025-01-01 00:00:00')"
            )
        legacy_engine.dispose()

        database.initialize_database(f"sqlite:///{legacy_path.as_posix()}", str(Path(self.temp_dir.name) / "empty"))

        columns = {column["name"] for column in inspect(database._engine).get_columns("user_lesson_progress")}
        self.assertTrue({"current_section", "last_activity", "active_quiz_session", "exercise_completed"}.issubset(columns))
        sign_columns = {column["name"] for column in inspect(database._engine).get_columns("signs")}
        self.assertTrue({"usage_context", "movement_notes", "example_sentence", "validation_status", "reviewed_by", "reviewed_at", "created_at", "updated_at"}.issubset(sign_columns))
        with database._session() as session:
            progress = session.get(database.UserLessonProgress, 7)
            attempt = session.get(database.QuizAttempt, 5)
            migrated_sign = session.get(database.Sign, 1)
            self.assertEqual((progress.username, progress.lesson_id, progress.status, progress.best_score, progress.attempts), ("legacy-learner", 12, "completed", 92.5, 2))
            self.assertEqual((attempt.username, attempt.lesson_id, attempt.score, attempt.percentage, attempt.passed), ("legacy-learner", 12, 9, 90, True))
            self.assertEqual(migrated_sign.usage_context, "Existing usage")
            self.assertEqual(migrated_sign.validation_status, "REVIEW_REQUIRED")
            self.assertEqual(migrated_sign.validation_notes, "Existing content has not been reviewed by a fluent sign-language reviewer.")


if __name__ == "__main__":
    unittest.main()