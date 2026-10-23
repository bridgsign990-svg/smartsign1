import tempfile
import unittest
from pathlib import Path

from sqlalchemy import select

import app
import database
from learning_mode import ensure_additional_alphabet_lessons, seed_starter_course, sync_avatar_catalog
from tools.audit_avatar_content import build_gap_report, build_report, collect_data


class TestAvatarContent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "avatar-content.sqlite3"
        self.empty_data_dir = Path(self.temp_dir.name) / "empty"
        database.initialize_database(f"sqlite:///{self.database_path.as_posix()}", str(self.empty_data_dir))
        seed_starter_course()
        self.catalog_summary = sync_avatar_catalog()
        self.extra_lessons = ensure_additional_alphabet_lessons()
        database.create_user("content-learner", "test-hash")
        self.client = app.app.test_client()
        with self.client.session_transaction() as browser_session:
            browser_session["username"] = "content-learner"

    def tearDown(self):
        if database._engine is not None:
            database._engine.dispose()
        self.temp_dir.cleanup()

    def sign(self, name):
        with database._session() as session:
            return session.scalar(select(database.Sign).where(database.Sign.name == name))

    def asset(self, sign_name, asset_type="SIGML"):
        with database._session() as session:
            sign = session.scalar(select(database.Sign).where(database.Sign.name == sign_name))
            return session.scalar(select(database.AvatarAsset).where(
                database.AvatarAsset.sign_id == sign.id,
                database.AvatarAsset.asset_type == asset_type,
            ))

    def test_catalog_maps_existing_assets_without_inventing_content(self):
        self.assertEqual(self.catalog_summary["sign_records"], 51)
        self.assertEqual(self.catalog_summary["asset_rows_synced"], 93)
        self.assertEqual(self.extra_lessons, 4)
        with database._session() as session:
            self.assertEqual(session.scalar(select(database.Course.id)), 1)
            self.assertEqual(session.query(database.Sign).count(), 51)
            self.assertEqual(session.query(database.AvatarAsset).count(), 93)
            self.assertEqual(session.query(database.Lesson).count(), 9)
            self.assertEqual(session.query(database.AvatarAsset).filter_by(validation_status="VALIDATED").count(), 0)
            self.assertEqual(session.query(database.Sign).filter_by(name="D").one().meaning, "")
            self.assertEqual(session.query(database.Sign).filter_by(name="D").one().validation_status, "UNVERIFIED")
            self.assertEqual(session.query(database.Sign).filter_by(name="HELLO").one().validation_status, "REVIEW_REQUIRED")

    def test_sync_and_seeding_are_idempotent_and_preserve_progress(self):
        lesson_id = self.client.get("/api/learning/courses").get_json()["courses"][0]["modules"][0]["lessons"][0]["id"]
        with database._session() as session, session.begin():
            session.add(database.UserLessonProgress(
                username="content-learner",
                lesson_id=lesson_id,
                status="review_required",
                progress_percentage=80,
                best_score=60,
                attempts=3,
                current_section="exercise",
            ))
        self.assertEqual(sync_avatar_catalog()["asset_rows_synced"], 93)
        self.assertEqual(ensure_additional_alphabet_lessons(), 0)
        seed_starter_course()
        with database._session() as session:
            progress = session.scalar(select(database.UserLessonProgress).where(
                database.UserLessonProgress.username == "content-learner",
                database.UserLessonProgress.lesson_id == lesson_id,
            ))
            self.assertEqual((progress.status, progress.progress_percentage, progress.best_score, progress.attempts), ("review_required", 80, 60, 3))
            self.assertEqual(session.query(database.Course).count(), 1)
            self.assertEqual(session.query(database.Lesson).count(), 9)
            self.assertEqual(session.query(database.AvatarAsset).count(), 93)

    def test_content_capabilities_and_validation_are_independent(self):
        course = self.client.get("/api/learning/courses").get_json()["courses"][0]
        lesson_ids = [lesson["id"] for module in course["modules"] for lesson in module["lessons"]]
        greetings = next(lesson_id for lesson_id in lesson_ids if self.client.get(f"/api/learning/lessons/{lesson_id}").get_json().get("title") == "Greetings")
        lesson = self.client.get(f"/api/learning/lessons/{greetings}").get_json()
        hello = next(sign for sign in lesson["signs"] if sign["name"] == "HELLO")
        letter = self.client.get(f"/api/learning/signs/{self.sign('A').id}").get_json()

        self.assertTrue(hello["avatar_available"])
        # HELLO is a motion sign: practised from a short clip by the gesture
        # model when one is installed, independent of avatar/validation status.
        gesture_ready = "HELLO" in app._recognition_labels()
        self.assertEqual(hello["recognition_available"], gesture_ready)
        self.assertEqual(hello["practice_available"], gesture_ready)
        self.assertEqual(hello["practice_mode"], "motion" if gesture_ready else "static")
        self.assertEqual(letter["practice_mode"], "static")
        self.assertEqual(hello["validation_status"], "REVIEW_REQUIRED")
        self.assertEqual(hello["avatar_asset"]["load_status"], "ASSET_LOADABLE")
        self.assertEqual(hello["avatar_asset"]["path"], "/static/avatar/sigml/asl/HELLO.sigml")
        self.assertTrue(hello["image_available"])
        self.assertEqual(hello["image_url"], "/static/signs/sign_hello.png")
        self.assertTrue(letter["recognition_available"])
        self.assertTrue(letter["avatar_available"])
        self.assertFalse(letter["validated"])
        self.assertEqual(letter["validation_status"], "REVIEW_REQUIRED")
        self.assertTrue(letter["movement_notes"] == "")

    def test_missing_and_broken_avatar_assets_degrade_gracefully(self):
        hello = self.sign("HELLO")
        with database._session() as session, session.begin():
            asset = session.scalar(select(database.AvatarAsset).where(
                database.AvatarAsset.sign_id == hello.id,
                database.AvatarAsset.path == "/static/avatar/sigml/asl/HELLO.sigml",
            ))
            session.delete(asset)
        sign = self.client.get(f"/api/learning/signs/{hello.id}").get_json()
        self.assertFalse(sign["avatar_available"])
        self.assertTrue(sign["image_available"])

        with database._session() as session, session.begin():
            session.add(database.AvatarAsset(
                sign_id=hello.id,
                asset_type="SIGML",
                path="/static/avatar/sigml/asl/REMOVED.sigml",
                format="SiGML",
                asset_status="AVAILABLE",
                validation_status="UNVERIFIED",
                load_status="ASSET_LOADABLE",
            ))
        sync_avatar_catalog()
        with database._session() as session:
            stale = session.scalar(select(database.AvatarAsset).where(database.AvatarAsset.path == "/static/avatar/sigml/asl/REMOVED.sigml"))
            self.assertEqual((stale.asset_status, stale.load_status), ("MISSING", "MISSING"))

        with database._session() as session, session.begin():
            asset = session.scalar(select(database.AvatarAsset).where(
                database.AvatarAsset.sign_id == hello.id,
                database.AvatarAsset.asset_type == "VRM_CLIP",
            ))
            asset.asset_status = "BROKEN"
            asset.load_status = "BROKEN"
        sign = self.client.get(f"/api/learning/signs/{hello.id}").get_json()
        self.assertTrue(sign["avatar_available"])
        self.assertTrue(sign["image_available"])

        with database._session() as session, session.begin():
            asset = session.scalar(select(database.AvatarAsset).where(
                database.AvatarAsset.sign_id == hello.id,
                database.AvatarAsset.path == "/static/avatar/sigml/asl/HELLO.sigml",
            ))
            asset.asset_status = "BROKEN"
            asset.load_status = "BROKEN"
        sign = self.client.get(f"/api/learning/signs/{hello.id}").get_json()
        self.assertFalse(sign["avatar_available"])

    def test_signer_review_state_survives_catalog_refresh_and_content_is_read_only(self):
        hello = self.sign("HELLO")
        with database._session() as session, session.begin():
            sign = session.get(database.Sign, hello.id)
            sign.validation_status = "VALIDATED"
            sign.reviewed_by = "Fixture reviewer"
            asset = session.scalar(select(database.AvatarAsset).where(
                database.AvatarAsset.sign_id == hello.id,
                database.AvatarAsset.asset_type == "SIGML",
            ))
            asset.validation_status = "VALIDATED"
        sync_avatar_catalog()
        response = self.client.get(f"/api/learning/signs/{hello.id}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["validated"])
        self.assertEqual(response.get_json()["reviewed_by"], "Fixture reviewer")
        self.assertEqual(self.client.post(f"/api/learning/signs/{hello.id}", json={"meaning": "changed"}).status_code, 405)
        self.assertEqual(self.sign("HELLO").meaning, "A greeting")

    def test_lesson_mappings_are_valid_and_unmapped_signs_are_reported(self):
        with database._session() as session:
            lessons = session.scalars(select(database.Lesson)).all()
            signs = session.scalars(select(database.Sign)).all()
            links = session.scalars(select(database.LessonSign)).all()
            sign_ids = {sign.id for sign in signs}
            lesson_ids = {lesson.id for lesson in lessons}
            pairs = [(link.lesson_id, link.sign_id) for link in links]
            self.assertTrue(all(lesson_id in lesson_ids and sign_id in sign_ids for lesson_id, sign_id in pairs))
            self.assertEqual(len(pairs), len(set(pairs)))
            self.assertTrue(all(any(link.lesson_id == lesson.id for link in links) for lesson in lessons))
            linked_sign_ids = {link.sign_id for link in links}
            unmapped = {sign.name for sign in signs if sign.id not in linked_sign_ids}

        self.assertEqual(len(unmapped), 17)
        self.assertIn("AMBULANCE", unmapped)
        self.assertIn("WHERE", unmapped)

    def test_gap_report_separates_completeness_from_validation_and_recognition(self):
        data = collect_data()
        report = build_gap_report(data)

        self.assertIn("| COMPLETE / PARTIAL / INCOMPLETE | 0 / 50 / 1 |", report)
        self.assertIn("| Dynamic gesture recognition | Not implemented |", report)
        self.assertIn("Static A-Z recognition", report)
        self.assertIn("No new lesson mappings were added.", report)
        self.assertIn("Requires curriculum/content review.", report)
        self.assertIn("AMBULANCE", report)
        self.assertIn("GOODBYE", report)
        self.assertIn("Technical render success, asset presence", report)
        self.assertIn("## Sign Inventory", report)
        self.assertIn("## Lesson Coverage", report)

    def test_report_covers_mapping_and_honestly_separates_render_from_review(self):
        data = collect_data()
        report = build_report(data)
        self.assertIn("50/50", report)
        self.assertIn("0", report)
        self.assertIn("ASSET_LOADABLE", report)
        self.assertIn("UNVERIFIED", report)
        self.assertIn("GOODBYE", report)
        self.assertIn("HELLO", report)
        self.assertIn("NOT RUNTIME TESTED", report.upper())
        self.assertEqual(len(data["unmapped_asset_paths"]), 0)
        self.assertEqual(sum(not row["sign_record"] for row in data["sigml_records"]), 0)


if __name__ == "__main__":
    unittest.main()
