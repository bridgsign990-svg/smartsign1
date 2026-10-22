import json
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
from sqlalchemy import select

import database
import config


STARTER_COURSE = {
    "slug": "beginner-sign-communication",
    "title": "Foundations in Sign Communication",
    "description": "Prototype sign-language learning for everyday communication, courtesy phrases, and supported letter practice. This starter course is a teaching demo and not a validated MSL curriculum.",
    "level": "beginner",
    "modules": [
        {
            "title": "Foundations",
            "description": "Useful signs for opening and guiding everyday interactions.",
            "lessons": [
                {
                    "title": "Greetings",
                    "description": "Open a conversation with a greeting and a courteous request.",
                    "objective": "Recognize HELLO and PLEASE and choose when to use each.",
                    "introduction": "A greeting can make the start of a conversation more welcoming. Learn HELLO, then add PLEASE when making a polite request.",
                    "signs": ["HELLO", "PLEASE"],
                    "steps": ["Watch each avatar demonstration once at normal speed.", "Notice where the hands begin and how the movement finishes.", "Use the situation question to connect each sign to a real interaction."],
                    "exercise_prompt": "You want to ask someone politely to wait. Which sign adds courtesy to your request?",
                    "exercise_answer": "PLEASE",
                    "exercise_explanation": "PLEASE is used to make a request more courteous.",
                    "situations": {"HELLO": "You meet someone and want to greet them. Which sign begins the interaction?"},
                },
                {
                    "title": "Yes and No",
                    "description": "Respond clearly to simple questions and offers.",
                    "objective": "Distinguish the signs YES and NO in common responses.",
                    "introduction": "Short responses can carry a lot of meaning. Compare the avatar demonstrations, then decide which response fits each situation.",
                    "signs": ["YES", "NO"],
                    "steps": ["Watch YES and NO separately; do not rely on handshape alone.", "Compare the direction and rhythm of each movement.", "Choose the response that matches the question or offer."],
                    "exercise_prompt": "Someone asks whether you would like water, and you do not. Which sign answers clearly?",
                    "exercise_answer": "NO",
                    "exercise_explanation": "NO communicates a negative response to the offer.",
                    "situations": {"YES": "Someone asks whether you are ready, and you are. Which response fits?", "NO": "You decline an offer. Which response fits?"},
                },
                {
                    "title": "Thanks and Apologies",
                    "description": "Express gratitude and acknowledge a mistake.",
                    "objective": "Recognize THANK YOU and SORRY and match them to their meaning.",
                    "introduction": "THANK YOU expresses gratitude. SORRY acknowledges an apology. Watch both signs and consider the social context in which each is used.",
                    "signs": ["THANK_YOU", "SORRY"],
                    "steps": ["Watch the avatar perform each sign.", "Notice the hand's starting point and path.", "Choose the sign that matches gratitude or apology."],
                    "exercise_prompt": "A person helps you carry something. Which sign expresses gratitude?",
                    "exercise_answer": "THANK_YOU",
                    "exercise_explanation": "THANK YOU is an expression of gratitude.",
                    "situations": {"THANK_YOU": "Someone gives you a thoughtful gift. Which sign expresses gratitude?", "SORRY": "You accidentally interrupt someone and want to apologize. Which sign fits?"},
                },
                {
                    "title": "Ask for Help",
                    "description": "Use HELP and STOP in situations where clarity matters.",
                    "objective": "Recognize HELP and STOP and identify their practical use.",
                    "introduction": "HELP and STOP are practical signs that may be useful in urgent or everyday situations. Learn each sign and use it in context.",
                    "signs": ["HELP", "STOP"],
                    "steps": ["Watch the avatar demonstration at normal speed.", "Repeat the movement while keeping the sign in view.", "Review the meaning and choose an appropriate situation."],
                    "exercise_prompt": "You need assistance carrying a heavy box. Which sign communicates that need?",
                    "exercise_answer": "HELP",
                    "exercise_explanation": "HELP is used to ask for assistance.",
                    "situations": {"HELP": "You need assistance with a task. Which sign communicates that?", "STOP": "You need someone to halt an action. Which sign is appropriate?"},
                },
            ],
        },
        {
            "title": "Fingerspelling",
            "description": "Practice static handshapes that the current recognition model supports.",
            "lessons": [
                {
                    "title": "Letters A to C",
                    "description": "Compare and practice the first three fingerspelled letters.",
                    "objective": "Recognize and perform the static handshapes A, B, and C.",
                    "introduction": "Fingerspelling is useful for names and words without a known sign. This lesson focuses on three static handshapes recognized by the current BridgeSign model.",
                    "signs": ["A", "B", "C"],
                    "steps": ["Study the avatar handshape for each letter.", "Keep your hand clearly visible and hold each shape steady.", "Use camera practice for a model prediction and confidence-based feedback."],
                    "exercise_prompt": "Which letter has a rounded, C-shaped hand?",
                    "exercise_answer": "C",
                    "exercise_explanation": "The letter C uses a curved handshape.",
                    "situations": {},
                },
            ],
        },
    ],
}

SIGN_CONTENT = {
    "HELLO": ("A greeting", "Open hand near the temple, then move it outward in a greeting motion.", "Use to greet someone or open an interaction.", "greetings", None),
    "PLEASE": ("A courteous request", "Place a flat hand on the chest and move it in a circular motion.", "Use to make a request more polite.", "courtesy", None),
    "YES": ("An affirmative response", "Make a fist and move it in a small nodding motion.", "Use to agree or answer affirmatively.", "responses", None),
    "NO": ("A negative response", "Bring the extended fingers toward the thumb in a short closing motion.", "Use to decline or answer negatively.", "responses", None),
    "THANK_YOU": ("An expression of gratitude", "Start with a flat hand near the chin and move it forward.", "Use to express thanks.", "courtesy", None),
    "SORRY": ("An apology", "Make a fist and rub it in a small circle on the chest.", "Use to apologize or express regret.", "courtesy", None),
    "HELP": ("A request for assistance", "Place a thumb-up hand on the other palm and lift both hands slightly.", "Use to ask someone for assistance.", "safety", None),
    "STOP": ("A request to halt", "Bring a flat hand down onto the other open palm.", "Use to ask someone to stop an action.", "safety", None),
    "A": ("Fingerspelled letter A", "Close the fingers into a fist with the thumb resting alongside.", "Use when fingerspelling a name or word containing A.", "alphabet", "A"),
    "B": ("Fingerspelled letter B", "Hold the fingers straight together with the thumb folded across the palm.", "Use when fingerspelling a name or word containing B.", "alphabet", "B"),
    "C": ("Fingerspelled letter C", "Curve the fingers and thumb into a C-shaped hand.", "Use when fingerspelling a name or word containing C.", "alphabet", "C"),
}


def _now():
    return datetime.now(timezone.utc)


_SECTION_ORDER = {name: index for index, name in enumerate(("introduction", "demonstration", "explanation", "practice", "exercise", "test", "result"))}
_LAST_LEARNING_SECTION = _SECTION_ORDER["test"]


def _json(value):
    return json.loads(value or "[]")


def _static_recognition_labels():
    models_dir = Path(config.MODELS_DIR)
    for filename in ("sign_model_light_classes.json", "sign_model_classes.json"):
        path = models_dir / filename
        if path.is_file():
            try:
                labels = json.loads(path.read_text(encoding="utf-8"))
                return {str(label).upper() for label in labels}
            except (OSError, json.JSONDecodeError, TypeError):
                continue
    return set()


def gesture_recognition_labels():
    """Motion signs (words, plus J/Z) the trained gesture model can check from a short clip."""
    models_dir = Path(config.MODELS_DIR)
    for filename in ("gesture_lstm_v2_meta.json", "gesture_lstm_meta.json"):
        path = models_dir / filename
        if path.is_file():
            try:
                labels = json.loads(path.read_text(encoding="utf-8")).get("classes", [])
                return {str(label).upper() for label in labels if str(label).upper() != "STATIC"}
            except (OSError, json.JSONDecodeError, TypeError, AttributeError):
                continue
    return set()


def _recognition_labels():
    return _static_recognition_labels() | gesture_recognition_labels()


def _image_label(path):
    label = path.stem.removeprefix("sign_").upper()
    return "THANK_YOU" if label == "THANKYOU" else label


def _asset_load_status(asset_type, path, render_results):
    if asset_type == "SIGML":
        try:
            ET.parse(path)
        except (ET.ParseError, OSError):
            return "BROKEN"
        result = render_results.get(path.stem.upper())
        return "ASSET_LOADABLE" if result and result.get("ok") else "RENDER_FAILED" if result else "XML_PARSEABLE"
    if asset_type == "VRM_CLIP":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and float(payload.get("duration", 0)) > 0 and isinstance(payload.get("tracks"), list):
                return "STRUCTURE_VALID"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return "BROKEN"
    if asset_type == "IMAGE":
        try:
            with Image.open(path) as image:
                image.verify()
            return "IMAGE_DECODABLE"
        except (OSError, ValueError):
            return "BROKEN"
    return "NOT_TESTED"


def _upsert_avatar_asset(session, sign, path, asset_type, format_name, version, status, load_status, notes):
    relative_path = "/" + path.relative_to(Path(config.BASE_DIR)).as_posix()
    asset = session.scalar(select(database.AvatarAsset).where(database.AvatarAsset.path == relative_path))
    values = {
        "sign_id": sign.id if sign else None,
        "asset_type": asset_type,
        "format": format_name,
        "language": "ASL",
        "version": version,
        "asset_status": status,
        "load_status": load_status,
        "notes": notes,
    }
    if asset is None:
        session.add(database.AvatarAsset(path=relative_path, validation_status="UNVERIFIED", **values))
    else:
        for key, value in values.items():
            setattr(asset, key, value)
        if asset.validation_status not in {"VALIDATED", "REJECTED", "DEPRECATED", "REVIEW_REQUIRED"}:
            asset.validation_status = "UNVERIFIED"
        asset.updated_at = _now()


def sync_avatar_catalog():
    root = Path(config.BASE_DIR)
    sigml_dir = root / "static" / "avatar" / "sigml" / "asl"
    clip_dir = root / "static" / "avatar" / "animations"
    image_dir = root / "static" / "signs"
    model_path = root / "static" / "avatar" / "model" / "avatar.vrm"
    report_path = root / "cwasa_sigml_validation_report.json"
    try:
        render_results = {
            item["label"].upper(): item
            for item in json.loads(report_path.read_text(encoding="utf-8")).get("results", [])
        } if report_path.is_file() else {}
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        render_results = {}

    sigml_paths = {path.stem.upper(): path for path in sigml_dir.glob("*.sigml")}
    clip_paths = {path.stem.upper(): path for path in clip_dir.glob("*.json")}
    image_paths = {_image_label(path): path for path in image_dir.iterdir() if path.is_file()}
    labels = set(sigml_paths) | set(clip_paths) | set(image_paths)
    recognition_labels = _recognition_labels()
    updated = 0

    with database._session() as session, session.begin():
        signs = {sign.name.upper(): sign for sign in session.scalars(select(database.Sign)).all()}
        for label in sorted(labels):
            sign = signs.get(label)
            if sign is None:
                sign = database.Sign(
                    name=label,
                    meaning="",
                    description="",
                    usage="",
                    usage_context="",
                    category="unclassified",
                    difficulty="unspecified",
                    content_type="LETTER" if len(label) == 1 else "WORD",
                    validation_status="UNVERIFIED",
                    validation_notes="An avatar or image file exists; reviewed sign content has not been provided.",
                    common_mistakes_json="[]",
                )
                session.add(sign)
                session.flush()
                signs[label] = sign

            sign.content_type = "LETTER" if label in _static_recognition_labels() and len(label) == 1 else sign.content_type or "WORD"
            if label in recognition_labels:
                sign.recognition_label = label
            sigml_path = sigml_paths.get(label)
            if sigml_path:
                sign.avatar_animation = sign.avatar_animation or f"sigml:{label}"
            image_path = image_paths.get(label)
            if image_path:
                sign.image_url = sign.image_url or "/" + image_path.relative_to(root).as_posix()
            if not sign.usage_context and sign.usage:
                sign.usage_context = sign.usage
            if sign.validation_status == "UNVERIFIED" and (sign.meaning or sign.description or sign.usage_context):
                sign.validation_status = "REVIEW_REQUIRED"
                if not sign.validation_notes:
                    sign.validation_notes = "Existing content has not been reviewed by a fluent sign-language reviewer."
            sign.updated_at = _now()

            if sigml_path:
                load_status = _asset_load_status("SIGML", sigml_path, render_results)
                asset_status = "BROKEN" if load_status in {"BROKEN", "RENDER_FAILED"} else "DRAFT"
                _upsert_avatar_asset(
                    session, sign, sigml_path, "SIGML", "SiGML", "draft-asl-dictionary",
                    asset_status, load_status,
                    "Repository SiGML README marks this dictionary as draft; renderer success is not linguistic validation.",
                )
                updated += 1
            clip_path = clip_paths.get(label)
            if clip_path:
                load_status = _asset_load_status("VRM_CLIP", clip_path, render_results)
                _upsert_avatar_asset(
                    session, sign, clip_path, "VRM_CLIP", "JSON", "motion-recorder-clip",
                    "AVAILABLE" if load_status == "STRUCTURE_VALID" else "BROKEN", load_status,
                    "Clip structure inspected only; VRM runtime playback and sign-language accuracy have not been reviewed.",
                )
                updated += 1
            if image_path:
                load_status = _asset_load_status("IMAGE", image_path, render_results)
                actual_format = "UNKNOWN"
                try:
                    with Image.open(image_path) as image:
                        actual_format = str(image.format or "UNKNOWN").upper()
                except (OSError, ValueError):
                    pass
                extension = image_path.suffix.lstrip(".").upper()
                notes = f"Stored file extension is {extension}; detected bytes are {actual_format}. Not signer-reviewed."
                _upsert_avatar_asset(
                    session, sign, image_path, "IMAGE", actual_format, "repository-sign-image",
                    "AVAILABLE" if load_status == "IMAGE_DECODABLE" else "BROKEN", load_status, notes,
                )
                updated += 1

        if model_path.is_file():
            _upsert_avatar_asset(
                session, None, model_path, "VRM_MODEL", "VRM", "repository-avatar-model",
                "AVAILABLE", "NOT_TESTED", "Shared avatar model file exists; not runtime-tested by this catalog sync.",
            )
            updated += 1
        current_paths = {
            "/" + path.relative_to(root).as_posix()
            for path in (*sigml_paths.values(), *clip_paths.values(), *image_paths.values())
        }
        if model_path.is_file():
            current_paths.add("/" + model_path.relative_to(root).as_posix())
        for asset in session.scalars(select(database.AvatarAsset)).all():
            if asset.path.startswith(("/static/avatar/sigml/", "/static/avatar/animations/", "/static/signs/", "/static/avatar/model/")) and asset.path not in current_paths:
                asset.asset_status = "MISSING"
                asset.load_status = "MISSING"
                asset.updated_at = _now()
    return {"sign_records": len(labels), "asset_rows_synced": updated}


def seed_starter_course():
    with database._session() as session, session.begin():
        course = session.scalar(select(database.Course).where(database.Course.slug == STARTER_COURSE["slug"]))
        if course:
            return

        course = database.Course(
            slug=STARTER_COURSE["slug"],
            title=STARTER_COURSE["title"],
            description=STARTER_COURSE["description"],
            level=STARTER_COURSE["level"],
        )
        session.add(course)
        session.flush()

        for sign_name, (meaning, description, usage, category, recognition_label) in SIGN_CONTENT.items():
            animation = f"sigml:{sign_name}"
            sign = session.scalar(select(database.Sign).where(database.Sign.name == sign_name))
            if sign is None:
                sign = database.Sign(
                    name=sign_name,
                    meaning=meaning,
                    description=description,
                    usage=usage,
                    avatar_animation=animation,
                    category=category,
                    recognition_label=recognition_label,
                    common_mistakes_json="[]",
                )
                session.add(sign)
                session.flush()

        for module_index, module_data in enumerate(STARTER_COURSE["modules"], start=1):
            module = database.Module(
                course_id=course.id,
                title=module_data["title"],
                description=module_data["description"],
                order_number=module_index,
            )
            session.add(module)
            session.flush()

            for lesson_index, lesson_data in enumerate(module_data["lessons"], start=1):
                lesson = database.Lesson(
                    module_id=module.id,
                    title=lesson_data["title"],
                    description=lesson_data["description"],
                    objective=lesson_data["objective"],
                    introduction=lesson_data["introduction"],
                    steps_json=json.dumps(lesson_data["steps"]),
                    exercise_prompt=lesson_data["exercise_prompt"],
                    exercise_options_json=json.dumps(lesson_data["signs"]),
                    exercise_answer=lesson_data["signs"].index(lesson_data["exercise_answer"]),
                    exercise_explanation=lesson_data["exercise_explanation"],
                    difficulty="beginner",
                    passing_score=70,
                    order_number=lesson_index,
                )
                session.add(lesson)
                session.flush()

                for sign_order, sign_name in enumerate(lesson_data["signs"], start=1):
                    sign = session.scalar(select(database.Sign).where(database.Sign.name == sign_name))
                    session.add(database.LessonSign(lesson_id=lesson.id, sign_id=sign.id, order_number=sign_order))
                    situations = lesson_data.get("situations", {})
                    if sign_name in situations:
                        session.add(database.QuizQuestion(
                            lesson_id=lesson.id,
                            question_type="situation",
                            prompt=situations[sign_name],
                            sign_id=sign.id,
                            options_json="[]",
                            answer_index=0,
                            explanation=sign.usage,
                            order_number=sign_order,
                        ))
                    session.add(database.QuizQuestion(
                        lesson_id=lesson.id,
                        question_type="avatar_identification" if sign.category != "alphabet" else "sign_recognition",
                        prompt="What sign is the avatar performing?" if sign.category != "alphabet" else "Which fingerspelled letter is shown?",
                        sign_id=sign.id,
                        options_json="[]",
                        answer_index=0,
                        explanation=f"{sign.name.replace('_', ' ')} means {sign.meaning.lower()}.",
                        order_number=len(lesson_data["signs"]) + sign_order,
                    ))
                    session.add(database.QuizQuestion(
                        lesson_id=lesson.id,
                        question_type="meaning_identification",
                        prompt="What does this sign communicate?" if sign.category != "alphabet" else "Which letter is this handshape?",
                        sign_id=sign.id,
                        options_json="[]",
                        answer_index=0,
                        explanation=f"{sign.name.replace('_', ' ')}: {sign.meaning}.",
                        order_number=2 * len(lesson_data["signs"]) + sign_order,
                    ))


def ensure_additional_alphabet_lessons():
    groups = (("D", "E", "F", "G", "H"), ("I", "J", "K", "L", "M"), ("N", "O", "P", "Q", "R"), ("S", "T", "U", "V", "W", "X", "Y", "Z"))
    with database._session() as session, session.begin():
        course = session.scalar(select(database.Course).where(database.Course.slug == STARTER_COURSE["slug"]))
        if course is None:
            return 0
        module = session.scalar(select(database.Module).where(
            database.Module.course_id == course.id,
            database.Module.title == "Fingerspelling",
        ))
        if module is None:
            next_order = (session.scalar(select(database.Module.order_number).where(database.Module.course_id == course.id).order_by(database.Module.order_number.desc()).limit(1)) or 0) + 1
            module = database.Module(
                course_id=course.id,
                title="Fingerspelling",
                description="Recognize and practice the static handshapes supported by BridgeSign's current letter model.",
                order_number=next_order,
            )
            session.add(module)
            session.flush()

        next_order = (session.scalar(select(database.Lesson.order_number).where(database.Lesson.module_id == module.id).order_by(database.Lesson.order_number.desc()).limit(1)) or 0) + 1
        created = 0
        for group in groups:
            title = f"Letters {group[0]} to {group[-1]}"
            lesson = session.scalar(select(database.Lesson).where(
                database.Lesson.module_id == module.id,
                database.Lesson.title == title,
            ))
            if lesson is not None:
                continue
            lesson = database.Lesson(
                module_id=module.id,
                title=title,
                description=f"Review the existing avatar and model labels for {', '.join(group)}.",
                objective=f"Recognize the fingerspelled letter labels {', '.join(group)}.",
                introduction="This lesson covers existing BridgeSign letter labels. The local avatar files are drafts, and written handshape or movement guidance has not been signer-reviewed.",
                steps_json=json.dumps([
                    "Select each letter to see the mapped avatar asset and its review status.",
                    "Use camera practice only when the listed recognition model supports the letter.",
                    "Treat confidence as a model estimate, not as signer validation.",
                ]),
                exercise_prompt=f"Which letter begins the group {', '.join(group)}?",
                exercise_options_json=json.dumps(group),
                exercise_answer=0,
                exercise_explanation=f"{group[0]} is the first letter in this lesson's group.",
                difficulty="beginner",
                passing_score=70,
                order_number=next_order,
            )
            session.add(lesson)
            session.flush()
            for position, letter in enumerate(group, start=1):
                sign = session.scalar(select(database.Sign).where(database.Sign.name == letter))
                if sign is None:
                    continue
                session.add(database.LessonSign(lesson_id=lesson.id, sign_id=sign.id, order_number=position))
                question_specs = (
                    ("avatar_identification", "What fingerspelled letter is the avatar performing?", position),
                    ("sign_recognition", "Perform the fingerspelled letter shown by the avatar.", len(group) + position),
                    ("meaning_identification", "Which letter is represented by this sign?", 2 * len(group) + position),
                )
                for question_type, prompt, question_order in question_specs:
                    session.add(database.QuizQuestion(
                        lesson_id=lesson.id,
                        question_type=question_type,
                        prompt=prompt,
                        sign_id=sign.id,
                        options_json="[]",
                        answer_index=0,
                        explanation=f"This question targets the model label {letter}; its sign content remains unverified.",
                        order_number=question_order,
                    ))
            next_order += 1
            created += 1
    return created


def _course_lessons(session, course_id):
    rows = session.execute(
        select(database.Module, database.Lesson)
        .join(database.Lesson, database.Lesson.module_id == database.Module.id)
        .where(database.Module.course_id == course_id)
        .order_by(database.Module.order_number, database.Lesson.order_number)
    ).all()
    return rows


def _access_status(rows, progress_by_lesson, lesson_id):
    previous_complete = True
    for _, lesson in rows:
        progress = progress_by_lesson.get(lesson.id)
        if lesson.id == lesson_id:
            if progress and progress.status == "completed":
                return "completed"
            if not previous_complete:
                return "locked"
            if progress and progress.status == "review_required":
                return "review_required"
            return "in_progress" if progress else "unlocked"
        previous_complete = bool(progress and progress.status == "completed")
    return None


def get_lesson_access(username, lesson_id):
    with database._session() as session:
        lesson = session.get(database.Lesson, lesson_id)
        if lesson is None:
            return None
        module = session.get(database.Module, lesson.module_id)
        rows = _course_lessons(session, module.course_id)
        progress = {
            row.lesson_id: row
            for row in session.scalars(select(database.UserLessonProgress).where(database.UserLessonProgress.username == username))
        }
        status = _access_status(rows, progress, lesson_id)
        return {"status": status, "allowed": status in {"unlocked", "in_progress", "review_required", "completed"}}


def _lesson_signs(session, lesson_id):
    return session.scalars(
        select(database.Sign)
        .join(database.LessonSign, database.LessonSign.sign_id == database.Sign.id)
        .where(database.LessonSign.lesson_id == lesson_id)
        .order_by(database.LessonSign.order_number)
    ).all()


def _sign_payload(session, sign):
    assets = session.scalars(
        select(database.AvatarAsset)
        .where(database.AvatarAsset.sign_id == sign.id)
        .order_by(database.AvatarAsset.asset_type, database.AvatarAsset.path)
    ).all()
    asset_payloads = [
        {
            "id": asset.id,
            "type": asset.asset_type,
            "path": asset.path,
            "format": asset.format,
            "language": asset.language,
            "version": asset.version,
            "status": asset.asset_status,
            "validation_status": asset.validation_status,
            "load_status": asset.load_status,
            "notes": asset.notes,
        }
        for asset in assets
    ]
    sigml = next((asset for asset in assets if asset.asset_type == "SIGML"), None)
    image = next((asset for asset in assets if asset.asset_type == "IMAGE"), None)
    video = next((asset for asset in assets if asset.asset_type == "VIDEO"), None)
    recognition_available = bool(sign.recognition_label and sign.recognition_label.upper() in _recognition_labels())
    avatar_available = bool(
        sigml
        and sigml.asset_status not in {"MISSING", "BROKEN", "UNSUPPORTED"}
        and sigml.load_status in {"ASSET_LOADABLE", "XML_PARSEABLE"}
    )
    image_available = bool(image and image.asset_status not in {"MISSING", "BROKEN", "UNSUPPORTED"} and image.load_status == "IMAGE_DECODABLE")
    video_available = bool(video and video.asset_status not in {"MISSING", "BROKEN", "UNSUPPORTED"})
    return {
        "id": sign.id,
        "name": sign.name,
        "display_name": sign.name.replace("_", " "),
        "meaning": sign.meaning,
        "description": sign.description,
        "usage": sign.usage_context or sign.usage,
        "usage_context": sign.usage_context or sign.usage,
        "movement_notes": sign.movement_notes,
        "example_sentence": sign.example_sentence,
        "category": sign.category,
        "difficulty": sign.difficulty,
        "content_type": sign.content_type,
        "recognition_label": sign.recognition_label,
        "recognition_available": recognition_available,
        "practice_available": recognition_available,
        # Words are checked from a short clip by the motion model; letters
        # (including the static J/Z handshapes) from a single snapshot.
        "practice_mode": (
            "motion"
            if recognition_available
            and len(sign.recognition_label) > 1
            and sign.recognition_label.upper() in gesture_recognition_labels()
            else "static"
        ),
        "avatar_animation": sign.avatar_animation,
        "avatar_available": avatar_available,
        "avatar_asset": asset_payloads[[item.asset_type for item in assets].index("SIGML")] if sigml else None,
        "image_available": image_available,
        "image_url": image.path if image_available else None,
        "video_available": video_available,
        "video_url": video.path if video_available else None,
        "validation_status": sign.validation_status,
        "validation_notes": sign.validation_notes,
        "reviewed_by": sign.reviewed_by,
        "reviewed_at": sign.reviewed_at.isoformat() if sign.reviewed_at else None,
        "validated": sign.validation_status == "VALIDATED",
        "assets": asset_payloads,
        "common_mistakes": _json(sign.common_mistakes_json),
    }


def _lesson_summary(session, username, module, lesson, progress, status):
    signs = _lesson_signs(session, lesson.id)
    return {
        "id": lesson.id,
        "title": lesson.title,
        "description": lesson.description,
        "order": lesson.order_number,
        "passing_score": lesson.passing_score,
        "status": status,
        "progress": progress.progress_percentage if progress else 0,
        "best_score": progress.best_score if progress else None,
        "sign_count": len(signs),
        "sign_ids": [sign.id for sign in signs],
    }


def _course_payload(session, username, course):
    rows = _course_lessons(session, course.id)
    progress_by_lesson = {
        row.lesson_id: row
        for row in session.scalars(select(database.UserLessonProgress).where(database.UserLessonProgress.username == username))
    }
    lessons = []
    modules = []
    completed = 0
    current = None
    for module, lesson in rows:
        status = _access_status(rows, progress_by_lesson, lesson.id)
        progress = progress_by_lesson.get(lesson.id)
        summary = _lesson_summary(session, username, module, lesson, progress, status)
        lessons.append((module, summary))
        if status == "completed":
            completed += 1
        if current is None and status in {"in_progress", "review_required", "unlocked"}:
            current = {"id": lesson.id, "title": lesson.title, "module": module.title, "status": status}

    for module in session.scalars(
        select(database.Module).where(database.Module.course_id == course.id).order_by(database.Module.order_number)
    ):
        module_lessons = [lesson for lesson_module, lesson in lessons if lesson_module.id == module.id]
        modules.append({
            "id": module.id,
            "title": module.title,
            "description": module.description,
            "order": module.order_number,
            "lessons": module_lessons,
        })

    total = len(lessons)
    return {
        "id": course.id,
        "slug": course.slug,
        "title": course.title,
        "description": course.description,
        "level": course.level,
        "progress": round(completed * 100 / total) if total else 0,
        "completed_lessons": completed,
        "total_lessons": total,
        "current_lesson": current,
        "modules": modules,
    }


def list_courses(username):
    with database._session() as session:
        courses = session.scalars(select(database.Course).order_by(database.Course.id)).all()
        return [_course_payload(session, username, course) for course in courses]


def get_course(username, course_id):
    with database._session() as session:
        course = session.get(database.Course, course_id)
        return _course_payload(session, username, course) if course else None


def get_lesson(username, lesson_id, section=None):
    with database._session() as session, session.begin():
        lesson = session.get(database.Lesson, lesson_id)
        if lesson is None:
            return None
        module = session.get(database.Module, lesson.module_id)
        rows = _course_lessons(session, module.course_id)
        progress_by_lesson = {
            row.lesson_id: row
            for row in session.scalars(select(database.UserLessonProgress).where(database.UserLessonProgress.username == username))
        }
        status = _access_status(rows, progress_by_lesson, lesson.id)
        if status == "locked":
            return {"locked": True, "status": status}
        progress = progress_by_lesson.get(lesson.id)
        if progress is None:
            progress = database.UserLessonProgress(
                username=username,
                lesson_id=lesson.id,
                status="in_progress",
                current_section="introduction",
                last_activity=_now(),
            )
            session.add(progress)
            session.flush()
        elif progress.status != "completed":
            progress.last_activity = _now()
        signs = _lesson_signs(session, lesson.id)
        return {
            "id": lesson.id,
            "title": lesson.title,
            "description": lesson.description,
            "objective": lesson.objective,
            "introduction": lesson.introduction,
            "steps": _json(lesson.steps_json),
            "passing_score": lesson.passing_score,
            "module": module.title,
            "status": status,
            "current_section": progress.current_section,
            "active_quiz_session": progress.active_quiz_session,
            "progress_percentage": progress.progress_percentage,
            "exercise_completed": progress.exercise_completed,
            "exercise": {
                "prompt": lesson.exercise_prompt,
                "options": _json(lesson.exercise_options_json),
            },
            "signs": [
                _sign_payload(session, sign) for sign in signs
            ],
        }


def get_sign(sign_id):
    with database._session() as session:
        sign = session.get(database.Sign, sign_id)
        if sign is None:
            return None
        return _sign_payload(session, sign)


def advance_lesson_section(username, lesson_id, section):
    if section not in _SECTION_ORDER or section == "result":
        return "invalid"
    with database._session() as session, session.begin():
        lesson = session.get(database.Lesson, lesson_id)
        if lesson is None:
            return "missing"
        progress = session.scalar(select(database.UserLessonProgress).where(
            database.UserLessonProgress.username == username,
            database.UserLessonProgress.lesson_id == lesson_id,
        ))
        if progress is None:
            progress = database.UserLessonProgress(
                username=username,
                lesson_id=lesson_id,
                status="in_progress",
                current_section="introduction",
            )
            session.add(progress)
            session.flush()
        target_index = _SECTION_ORDER[section]
        if progress.status == "completed" and section == "test":
            return "completed"
        max_reached = round(progress.progress_percentage * _LAST_LEARNING_SECTION / 100)
        if target_index > max_reached + 1:
            return "out_of_order"
        if section == "test" and not progress.exercise_completed:
            return "exercise_required"
        progress.current_section = section
        section_percentage = round(target_index * 100 / _LAST_LEARNING_SECTION)
        progress.progress_percentage = max(progress.progress_percentage, section_percentage)
        progress.last_activity = _now()
        return "ok"


def submit_exercise(username, lesson_id, answer_index):
    with database._session() as session, session.begin():
        lesson = session.get(database.Lesson, lesson_id)
        if lesson is None:
            return None
        access = get_lesson_access(username, lesson_id)
        if not access or not access["allowed"]:
            return {"locked": True}
        progress = session.scalar(select(database.UserLessonProgress).where(
            database.UserLessonProgress.username == username,
            database.UserLessonProgress.lesson_id == lesson_id,
        ))
        if progress is None or progress.current_section != "exercise":
            return {"out_of_order": True}
        options = _json(lesson.exercise_options_json)
        if not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
            return {"invalid": True}
        correct = answer_index == lesson.exercise_answer
        progress.exercise_completed = True
        progress.progress_percentage = max(progress.progress_percentage, round(_SECTION_ORDER["exercise"] * 100 / _LAST_LEARNING_SECTION))
        progress.last_activity = _now()
        return {
            "correct": correct,
            "feedback": "That is the right choice." if correct else "Not quite. Review the sign meaning, then try the final test when ready.",
            "explanation": lesson.exercise_explanation,
        }


def _question_payload(session, question, course_id):
    sign = session.get(database.Sign, question.sign_id) if question.sign_id else None
    if sign is None:
        return None
    course_signs = session.scalars(
        select(database.Sign)
        .join(database.LessonSign, database.LessonSign.sign_id == database.Sign.id)
        .join(database.Lesson, database.Lesson.id == database.LessonSign.lesson_id)
        .join(database.Module, database.Module.id == database.Lesson.module_id)
        .where(database.Module.course_id == course_id)
        .distinct()
    ).all()
    similar = [item for item in course_signs if item.category == sign.category and item.id != sign.id]
    distractors = similar + [item for item in course_signs if item.category != sign.category and item.id != sign.id]
    chosen = distractors[:3]
    options = [sign, *chosen]
    secrets.SystemRandom().shuffle(options)
    display_meanings = question.question_type == "meaning_identification"
    sign_content = _sign_payload(session, sign)
    avatar_available = question.question_type != "situation" and sign_content["avatar_available"]
    return {
        "id": question.id,
        "type": question.question_type,
        "prompt": question.prompt,
        "sign_id": sign.id,
        "avatar_label": sign.name,
        "avatar_asset": sign_content["avatar_asset"],
        "avatar_available": avatar_available,
        "recognition_label": sign.recognition_label,
        "options": [{"label": item.meaning if display_meanings else item.name.replace("_", " ")} for item in options],
        "option_sign_ids": [item.id for item in options],
        "answer_index": options.index(sign),
        "explanation": question.explanation,
    }


def start_quiz(username, lesson_id):
    with database._session() as session, session.begin():
        lesson = session.get(database.Lesson, lesson_id)
        if lesson is None:
            return None
        module = session.get(database.Module, lesson.module_id)
        access = get_lesson_access(username, lesson_id)
        if not access or not access["allowed"]:
            return {"locked": True}
        questions = session.scalars(
            select(database.QuizQuestion).where(database.QuizQuestion.lesson_id == lesson_id).order_by(database.QuizQuestion.order_number)
        ).all()
        payloads = [payload for question in questions if (payload := _question_payload(session, question, module.course_id))]
        if not payloads:
            return {"empty": True}
        attempt_number = len(session.scalars(
            select(database.QuizAttempt.id).where(
                database.QuizAttempt.username == username,
                database.QuizAttempt.lesson_id == lesson_id,
            )
        ).all()) + 1
        progress = session.scalar(select(database.UserLessonProgress).where(
            database.UserLessonProgress.username == username,
            database.UserLessonProgress.lesson_id == lesson_id,
        ))
        if progress is None:
            progress = database.UserLessonProgress(username=username, lesson_id=lesson_id, status="in_progress")
            session.add(progress)
        if (progress.current_section != "test" and progress.status != "review_required") or not progress.exercise_completed:
            return {"locked": True}
        token = str(uuid.uuid4())
        state = [{key: value for key, value in payload.items() if key != "explanation"} for payload in payloads]
        quiz_session = database.QuizSession(
            id=token,
            username=username,
            lesson_id=lesson_id,
            questions_json=json.dumps(state),
            answers_json="{}",
            attempt_number=attempt_number,
            started_at=_now(),
        )
        session.add(quiz_session)
        progress.active_quiz_session = token
        progress.current_section = "test"
        progress.status = "in_progress"
        progress.last_activity = _now()
        return {
            "session_id": token,
            "attempt_number": attempt_number,
            "passing_score": lesson.passing_score,
            "questions": [_public_question(payload) for payload in payloads],
        }


def answer_quiz(username, quiz_session_id, question_id, option_index):
    with database._session() as session, session.begin():
        quiz_session = session.get(database.QuizSession, quiz_session_id)
        if quiz_session is None or quiz_session.username != username:
            return {"missing": True}
        if quiz_session.completed:
            return {"completed": True}
        if not isinstance(option_index, int):
            return {"invalid": True}
        questions = _json(quiz_session.questions_json)
        question = next((item for item in questions if item["id"] == question_id), None)
        answers = json.loads(quiz_session.answers_json or "{}")
        if question is None or str(question_id) in answers or not 0 <= option_index < len(question["option_sign_ids"]):
            return {"invalid": True}
        if question["type"] == "sign_recognition":
            return {"recognition_required": True}
        correct = question["option_sign_ids"][option_index] == question["option_sign_ids"][question["answer_index"]]
        answers[str(question_id)] = {"selected_index": option_index, "correct": correct}
        quiz_session.answers_json = json.dumps(answers)
        return {"correct": correct, "feedback": "Correct. Keep going." if correct else "Not quite. Keep going; you will see the review at the end."}


def answer_quiz_prediction(username, quiz_session_id, question_id, predicted_label, confidence):
    with database._session() as session, session.begin():
        quiz_session = session.get(database.QuizSession, quiz_session_id)
        if quiz_session is None or quiz_session.username != username:
            return {"missing": True}
        if quiz_session.completed:
            return {"completed": True}
        questions = _json(quiz_session.questions_json)
        question = next((item for item in questions if item["id"] == question_id), None)
        answers = json.loads(quiz_session.answers_json or "{}")
        if question is None or question["type"] != "sign_recognition" or str(question_id) in answers:
            return {"invalid": True}
        sign_id = question["option_sign_ids"][question["answer_index"]]
        sign = session.get(database.Sign, sign_id)
        target = (sign.recognition_label or sign.name).upper()
        prediction = str(predicted_label or "").upper()
        confidence = max(0.0, min(1.0, float(confidence or 0)))
        correct = prediction == target and confidence >= 0.60
        option_index = next(
            (index for index, option_sign_id in enumerate(question["option_sign_ids"])
             if (session.get(database.Sign, option_sign_id).recognition_label or session.get(database.Sign, option_sign_id).name).upper() == prediction),
            -1,
        )
        answers[str(question_id)] = {
            "selected_index": option_index,
            "correct": correct,
            "predicted_label": prediction,
        }
        quiz_session.answers_json = json.dumps(answers)
        _update_mastery(session, username, sign_id, correct, confidence)
        session.add(database.PracticeAttempt(
            username=username,
            lesson_id=quiz_session.lesson_id,
            sign_id=sign_id,
            target_label=target,
            predicted_label=prediction,
            confidence=confidence,
            correct=correct,
            feedback="Recognized by the current model." if correct else "The current model did not confirm the target sign.",
        ))
        feedback = "Excellent! The model recognized the target handshape." if correct and confidence >= 0.90 else "Good recognition. Keep practicing the handshape." if correct else "Not quite. Hold the handshape clearly and try again."
        return {"correct": correct, "prediction": prediction or "No clear sign", "confidence": round(confidence, 3), "feedback": feedback}


def quiz_recognition_target(username, quiz_session_id, question_id):
    with database._session() as session:
        quiz_session = session.get(database.QuizSession, quiz_session_id)
        if quiz_session is None or quiz_session.username != username or quiz_session.completed:
            return None
        question = next((item for item in _json(quiz_session.questions_json) if item["id"] == question_id), None)
        if question is None or question["type"] != "sign_recognition":
            return None
        sign_id = question["option_sign_ids"][question["answer_index"]]
        sign = session.get(database.Sign, sign_id)
        return {"lesson_id": quiz_session.lesson_id, "sign_id": sign_id, "recognition_label": sign.recognition_label or sign.name}


def _public_question(question):
    return {
        key: value
        for key, value in question.items()
        if key not in {"option_sign_ids", "answer_index", "explanation", "avatar_label", "recognition_label", "sign_id", "avatar_asset"}
    }


def quiz_demonstration(username, quiz_session_id, question_id):
    with database._session() as session:
        quiz_session = session.get(database.QuizSession, quiz_session_id)
        if quiz_session is None or quiz_session.username != username:
            return None
        question = next((item for item in _json(quiz_session.questions_json) if item["id"] == question_id), None)
        if question is None:
            return None
        if question.get("type") == "situation":
            return {"asset": None}
        sign = session.get(database.Sign, question.get("sign_id"))
        if sign is None:
            return {"asset": None}
        payload = _sign_payload(session, sign)
        if not payload["avatar_available"]:
            return {"asset": None}
        return {
            "asset": {
                "path": payload["avatar_asset"]["path"],
                "label": sign.name,
                "status": payload["avatar_asset"]["status"],
                "validation_status": sign.validation_status,
            }
        }


def _update_mastery(session, username, sign_id, correct, confidence=None):
    mastery = session.get(database.UserSignMastery, (username, sign_id))
    if mastery is None:
        mastery = database.UserSignMastery(username=username, sign_id=sign_id)
        session.add(mastery)
        session.flush()
    old_attempts = mastery.practice_attempts
    old_score = mastery.mastery_score
    score = (confidence * 100) if confidence is not None and correct else (100.0 if correct else 0.0)
    mastery.practice_attempts = old_attempts + 1
    mastery.correct_attempts += int(correct)
    mastery.mastery_score = round((old_score * old_attempts + score) / mastery.practice_attempts, 1)
    mastery.last_practiced = _now()


def complete_quiz(username, quiz_session_id):
    with database._session() as session, session.begin():
        quiz_session = session.get(database.QuizSession, quiz_session_id)
        if quiz_session is None or quiz_session.username != username:
            return {"missing": True}
        if quiz_session.completed:
            return {"completed": True}
        answers = json.loads(quiz_session.answers_json or "{}")
        question_states = _json(quiz_session.questions_json)
        if len(answers) != len(question_states):
            return {"incomplete": True, "answered": len(answers), "total": len(question_states)}
        lesson = session.get(database.Lesson, quiz_session.lesson_id)
        correct_count = sum(answer["correct"] for answer in answers.values())
        total = len(question_states)
        percentage = round(correct_count * 100 / total, 1) if total else 0
        passed = percentage >= lesson.passing_score
        session.add(database.QuizAttempt(
            username=username,
            lesson_id=lesson.id,
            score=correct_count,
            total_questions=total,
            percentage=percentage,
            passed=passed,
            attempt_number=quiz_session.attempt_number,
            started_at=quiz_session.started_at,
            completed_at=_now(),
        ))
        quiz_session.completed = True
        progress = session.scalar(select(database.UserLessonProgress).where(
            database.UserLessonProgress.username == username,
            database.UserLessonProgress.lesson_id == lesson.id,
        ))
        if progress is None:
            progress = database.UserLessonProgress(username=username, lesson_id=lesson.id)
            session.add(progress)
            session.flush()
        progress.attempts += 1
        progress.best_score = max(progress.best_score, percentage)
        progress.progress_percentage = 100 if passed else max(progress.progress_percentage, round(_SECTION_ORDER["test"] * 100 / _LAST_LEARNING_SECTION))
        progress.status = "completed" if passed else "review_required"
        progress.current_section = "result"
        progress.last_activity = _now()
        progress.active_quiz_session = None
        if passed and progress.completed_at is None:
            progress.completed_at = _now()

        review = []
        for question in question_states:
            answer = answers[str(question["id"])]
            sign_id = question["option_sign_ids"][question["answer_index"]]
            if question["type"] != "sign_recognition":
                _update_mastery(session, username, sign_id, answer["correct"])
            sign = session.get(database.Sign, sign_id)
            selected_index = answer["selected_index"]
            selected_label = (
                question["options"][selected_index]["label"]
                if selected_index >= 0
                else answer.get("predicted_label") or "No clear sign detected"
            )
            review.append({
                "prompt": question["prompt"],
                "selected": selected_label,
                "correct_answer": question["options"][question["answer_index"]]["label"],
                "correct": answer["correct"],
                "explanation": question["explanation"] if "explanation" in question else f"{sign.name.replace('_', ' ')}: {sign.meaning}.",
            })
        return {
            "score": correct_count,
            "total": total,
            "percentage": percentage,
            "passed": passed,
            "passing_score": lesson.passing_score,
            "lesson_id": lesson.id,
            "review": review,
        }


def resume_quiz(username, quiz_session_id):
    with database._session() as session:
        quiz_session = session.get(database.QuizSession, quiz_session_id)
        if quiz_session is None or quiz_session.username != username or quiz_session.completed:
            return None
        questions = _json(quiz_session.questions_json)
        answers = json.loads(quiz_session.answers_json or "{}")
        return {
            "session_id": quiz_session.id,
            "attempt_number": quiz_session.attempt_number,
            "questions": [_public_question(question) for question in questions],
            "answered_ids": [int(question_id) for question_id in answers],
        }


def record_practice(username, lesson_id, sign_id, predicted_label, confidence):
    with database._session() as session, session.begin():
        lesson = session.get(database.Lesson, lesson_id)
        sign = session.get(database.Sign, sign_id)
        access = get_lesson_access(username, lesson_id)
        if lesson is None or sign is None:
            return None
        if not access or not access["allowed"]:
            return {"locked": True}
        target = (sign.recognition_label or "").upper()
        prediction = (predicted_label or "").upper()
        confidence = max(0.0, min(1.0, float(confidence or 0)))
        matched = bool(target and prediction == target)
        correct = matched and confidence >= 0.60
        if correct and confidence >= 0.90:
            feedback = "Excellent! The sign matches the target with high confidence."
        elif correct and confidence >= 0.75:
            feedback = "Good recognition. Keep practicing to make the handshape consistent."
        elif matched:
            feedback = "The model recognized the target, but confidence is low. Hold the handshape clearly and try again."
        elif confidence >= 0.60:
            feedback = "You're close. Compare your handshape with the demonstration and try again."
        else:
            feedback = "Let's try again. Watch the demonstration and keep your hand clearly in frame."
        session.add(database.PracticeAttempt(
            username=username,
            lesson_id=lesson_id,
            sign_id=sign_id,
            target_label=target,
            predicted_label=prediction,
            confidence=confidence,
            correct=correct,
            feedback=feedback,
        ))
        _update_mastery(session, username, sign_id, correct, confidence)
        progress = session.scalar(select(database.UserLessonProgress).where(
            database.UserLessonProgress.username == username,
            database.UserLessonProgress.lesson_id == lesson_id,
        ))
        if progress is None:
            progress = database.UserLessonProgress(username=username, lesson_id=lesson_id, status="in_progress")
            session.add(progress)
        progress.current_section = "practice"
        progress.last_activity = _now()
        return {"target": target, "prediction": prediction, "confidence": round(confidence, 3), "correct": correct, "feedback": feedback}


def learning_progress(username):
    courses = list_courses(username)
    with database._session() as session:
        mastery_rows = session.execute(
            select(database.UserSignMastery, database.Sign)
            .join(database.Sign, database.Sign.id == database.UserSignMastery.sign_id)
            .where(database.UserSignMastery.username == username)
        ).all()
        mastered = [row.name for mastery, row in mastery_rows if mastery.mastery_score >= 90]
        needs_practice = [
            {"name": sign.name.replace("_", " "), "score": mastery.mastery_score, "sign_id": sign.id}
            for mastery, sign in mastery_rows if mastery.practice_attempts and mastery.mastery_score < 60
        ]
        recent = session.execute(
            select(database.QuizAttempt, database.Lesson)
            .join(database.Lesson, database.Lesson.id == database.QuizAttempt.lesson_id)
            .where(database.QuizAttempt.username == username)
            .order_by(database.QuizAttempt.completed_at.desc())
            .limit(5)
        ).all()
        return {
            "courses": courses,
            "mastered_signs": mastered,
            "needs_practice": needs_practice,
            "recent_tests": [
                {"lesson": lesson.title, "score": attempt.score, "total": attempt.total_questions, "percentage": attempt.percentage, "passed": attempt.passed, "completed_at": attempt.completed_at.isoformat()}
                for attempt, lesson in recent
            ],
        }


def sign_mastery(username):
    with database._session() as session:
        rows = session.execute(
            select(database.UserSignMastery, database.Sign)
            .join(database.Sign, database.Sign.id == database.UserSignMastery.sign_id)
            .where(database.UserSignMastery.username == username)
            .order_by(database.UserSignMastery.mastery_score.desc())
        ).all()
        result = []
        for mastery, sign in rows:
            score = mastery.mastery_score
            level = "Mastered" if score >= 90 else "Strong" if score >= 80 else "Developing" if score >= 60 else "Needs Practice"
            result.append({"sign_id": sign.id, "name": sign.name.replace("_", " "), "score": score, "level": level, "attempts": mastery.practice_attempts, "last_practiced": mastery.last_practiced.isoformat() if mastery.last_practiced else None})
        return result