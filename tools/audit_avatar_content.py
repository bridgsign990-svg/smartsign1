from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from sqlalchemy import func, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import database


ROOT = Path(config.BASE_DIR)
SIGML_DIR = ROOT / "static" / "avatar" / "sigml" / "asl"
CLIP_DIR = ROOT / "static" / "avatar" / "animations"
IMAGE_DIR = ROOT / "static" / "signs"
MODEL_PATH = ROOT / "static" / "avatar" / "model" / "avatar.vrm"
RENDER_REPORT_PATH = ROOT / "cwasa_sigml_validation_report.json"
OUTPUT_PATH = ROOT / "AVATAR_CONTENT_AUDIT.md"
GAP_REPORT_PATH = ROOT / "AVATAR_CONTENT_GAP_REPORT.md"
VALIDATION_STATES = {"UNVERIFIED", "REVIEW_REQUIRED", "VALIDATED", "REJECTED", "DEPRECATED"}
ASSET_STATES = {"AVAILABLE", "MISSING", "BROKEN", "DRAFT", "TEST_ONLY", "UNSUPPORTED"}


def display_label(label: str) -> str:
    return label.replace("_", " ")


def _safe_json_list(value: str | None) -> list:
    try:
        parsed = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def image_label(path: Path) -> str:
    label = path.stem.removeprefix("sign_").upper()
    return "THANK_YOU" if label == "THANKYOU" else label


def inspect_sigml(path: Path) -> str:
    try:
        ET.parse(path)
        return "XML_PARSEABLE"
    except (ET.ParseError, OSError):
        return "BROKEN"


def inspect_clip(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return "BROKEN_JSON"
        duration = float(payload.get("duration", 0))
        tracks = payload.get("tracks")
        if duration <= 0 or not isinstance(tracks, list):
            return "INCOMPLETE_JSON"
        return "STRUCTURE_VALID"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "BROKEN_JSON"


def inspect_image(path: Path) -> tuple[str, str]:
    try:
        with Image.open(path) as image:
            image_format = str(image.format or "UNKNOWN").upper()
            image.verify()
        return "IMAGE_DECODABLE", image_format
    except (OSError, ValueError):
        return "BROKEN_IMAGE", "UNKNOWN"


def load_model_classes() -> set[str]:
    for filename in ("sign_model_light_classes.json", "sign_model_classes.json"):
        path = ROOT / "models" / filename
        if path.is_file():
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
                return {str(value).upper() for value in values}
            except (OSError, json.JSONDecodeError, TypeError):
                continue
    return set()


def load_render_results() -> dict[str, dict]:
    if not RENDER_REPORT_PATH.is_file():
        return {}
    try:
        payload = json.loads(RENDER_REPORT_PATH.read_text(encoding="utf-8"))
        return {str(item.get("label", "")).upper(): item for item in payload.get("results", [])}
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return {}


def collect_data():
    if database._Session is None:
        database.initialize_database()
    with database._session() as session:
        signs = session.scalars(select(database.Sign).order_by(database.Sign.name)).all()
        lessons = session.scalars(select(database.Lesson).order_by(database.Lesson.id)).all()
        links = session.scalars(select(database.LessonSign)).all()
        lesson_sign_names = {}
        for link in links:
            lesson_sign_names.setdefault(link.lesson_id, []).append((link.order_number, link.sign_id))
        lesson_titles = {lesson.id: lesson.title for lesson in lessons}
        sign_lessons: dict[int, list[str]] = {}
        for link in links:
            sign_lessons.setdefault(link.sign_id, []).append(lesson_titles.get(link.lesson_id, str(link.lesson_id)))
        signs_by_name = {sign.name.upper(): sign for sign in signs}
        assets_model = getattr(database, "AvatarAsset", None)
        mapped_assets = {}
        asset_rows = []
        if assets_model is not None and "avatar_assets" in database.Base.metadata.tables:
            asset_rows = session.scalars(select(assets_model).order_by(assets_model.id)).all()
            mapped_assets = {asset.path: asset for asset in asset_rows}
        sign_names_by_id = {sign.id: sign.name for sign in signs}
        lesson_details = []
        for lesson in lessons:
            mapped_sign_ids = [sign_id for _, sign_id in sorted(lesson_sign_names.get(lesson.id, []))]
            lesson_details.append({
                "id": lesson.id,
                "title": lesson.title,
                "objective": lesson.objective,
                "introduction": lesson.introduction,
                "steps": _safe_json_list(lesson.steps_json),
                "exercise_prompt": lesson.exercise_prompt,
                "exercise_options": _safe_json_list(lesson.exercise_options_json),
                "exercise_answer": lesson.exercise_answer,
                "exercise_explanation": lesson.exercise_explanation,
                "sign_ids": mapped_sign_ids,
                "sign_names": [sign_names_by_id.get(sign_id, f"UNKNOWN:{sign_id}") for sign_id in mapped_sign_ids],
                "quiz_question_count": session.scalar(
                    select(func.count()).select_from(database.QuizQuestion).where(database.QuizQuestion.lesson_id == lesson.id)
                ),
            })

    render_results = load_render_results()
    model_classes = load_model_classes()
    sigml_paths = sorted(SIGML_DIR.glob("*.sigml"))
    clip_paths = sorted(CLIP_DIR.glob("*.json"))
    image_paths = sorted(IMAGE_DIR.glob("*"))
    image_paths = [path for path in image_paths if path.is_file()]
    records = []

    for path in sigml_paths:
        label = path.stem.upper()
        parse_status = inspect_sigml(path)
        render = render_results.get(label)
        records.append({
            "label": label,
            "type": "SIGML",
            "path": "/" + path.relative_to(ROOT).as_posix(),
            "format": path.suffix.lstrip(".").upper(),
            "file_status": parse_status,
            "load_status": "ASSET_LOADABLE" if render and render.get("ok") else "RENDER_FAILED" if render else "NOT_TESTED",
            "render_tested": bool(render),
            "render_passed": bool(render and render.get("ok")),
            "render_error": str(render.get("error", "")) if render else "",
            "sign_record": label in signs_by_name,
            "lessons": sign_lessons.get(signs_by_name[label].id, []) if label in signs_by_name else [],
            "recognition": label in model_classes,
            "validation_status": getattr(signs_by_name.get(label), "validation_status", None) or "UNVERIFIED",
            "asset_status": getattr(mapped_assets.get("/" + path.relative_to(ROOT).as_posix()), "asset_status", None) or "DRAFT",
            "draft_source": True,
        })

    for path in clip_paths:
        label = path.stem.upper()
        clip_status = inspect_clip(path)
        records.append({
            "label": label,
            "type": "VRM_CLIP",
            "path": "/" + path.relative_to(ROOT).as_posix(),
            "format": "JSON",
            "file_status": clip_status,
            "load_status": "NOT_RUNTIME_TESTED",
            "render_tested": False,
            "render_passed": False,
            "render_error": "",
            "sign_record": label in signs_by_name,
            "lessons": sign_lessons.get(signs_by_name[label].id, []) if label in signs_by_name else [],
            "recognition": label in model_classes,
            "validation_status": getattr(signs_by_name.get(label), "validation_status", None) or "UNVERIFIED",
            "asset_status": getattr(mapped_assets.get("/" + path.relative_to(ROOT).as_posix()), "asset_status", None) or "AVAILABLE",
            "draft_source": True,
        })

    for path in image_paths:
        label = image_label(path)
        file_status, actual_format = inspect_image(path)
        records.append({
            "label": label,
            "type": "IMAGE",
            "path": "/" + path.relative_to(ROOT).as_posix(),
            "format": actual_format,
            "file_status": file_status,
            "load_status": "IMAGE_DECODABLE" if file_status == "IMAGE_DECODABLE" else "BROKEN",
            "render_tested": False,
            "render_passed": False,
            "render_error": "",
            "sign_record": label in signs_by_name,
            "lessons": sign_lessons.get(signs_by_name[label].id, []) if label in signs_by_name else [],
            "recognition": label in model_classes,
            "validation_status": getattr(signs_by_name.get(label), "validation_status", None) or "UNVERIFIED",
            "asset_status": getattr(mapped_assets.get("/" + path.relative_to(ROOT).as_posix()), "asset_status", None) or "AVAILABLE",
            "draft_source": True,
        })

    letters = []
    for letter in sorted(model_classes.intersection({chr(value) for value in range(ord("A"), ord("Z") + 1)})):
        sign = signs_by_name.get(letter)
        sigml = next((record for record in records if record["type"] == "SIGML" and record["label"] == letter), None)
        image = next((record for record in records if record["type"] == "IMAGE" and record["label"] == letter), None)
        letters.append({
            "letter": letter,
            "recognition": True,
            "sign_record": sign is not None,
            "lesson": bool(sign and sign_lessons.get(sign.id)),
            "avatar": bool(sigml),
            "avatar_load_status": sigml["load_status"] if sigml else "MISSING",
            "image": bool(image),
            "video": False,
            "meaning": bool(sign and sign.meaning),
            "movement_notes": bool(getattr(sign, "movement_notes", None)) if sign else False,
            "usage_context": bool(getattr(sign, "usage_context", None) or (sign and sign.usage)) if sign else False,
            "validation_status": (getattr(sign, "validation_status", None) or "UNVERIFIED") if sign else None,
        })

    orphaned_assets = [record for record in records if record["type"] in {"SIGML", "VRM_CLIP", "IMAGE"} and not record["sign_record"]]
    missing_avatar_signs = [sign.name for sign in signs if not any(record["type"] == "SIGML" and record["label"] == sign.name.upper() and record["file_status"] != "BROKEN" for record in records)]
    linked_sign_ids = set(sign_lessons)
    sign_validation = [getattr(sign, "validation_status", None) or "UNVERIFIED" for sign in signs]
    image_records = [record for record in records if record["type"] == "IMAGE"]
    sigml_records = [record for record in records if record["type"] == "SIGML"]
    clip_records = [record for record in records if record["type"] == "VRM_CLIP"]
    missing_or_broken_assets = sum(record["file_status"] in {"BROKEN", "BROKEN_JSON", "BROKEN_IMAGE", "INCOMPLETE_JSON"} for record in records)
    lessons_with_avatar = sum(
        any(
            record["type"] == "SIGML" and record["label"] == sign.name.upper() and record["file_status"] != "BROKEN"
            for sign in signs if any(lesson.id == link.lesson_id and sign.id == link.sign_id for link in links)
            for record in records
        )
        for lesson in lessons
    )
    return {
        "signs": signs,
        "lessons": lessons,
        "links": links,
        "sign_lessons": sign_lessons,
        "lesson_details": lesson_details,
        "asset_rows": [
            {
                "id": asset.id,
                "sign_id": asset.sign_id,
                "type": asset.asset_type,
                "path": asset.path,
                "status": asset.asset_status,
                "load_status": asset.load_status,
                "validation_status": asset.validation_status,
            }
            for asset in asset_rows
        ],
        "signs_by_name": signs_by_name,
        "records": records,
        "letters": letters,
        "orphaned_assets": orphaned_assets,
        "missing_avatar_signs": missing_avatar_signs,
        "linked_sign_ids": linked_sign_ids,
        "validation_counts": {state: sign_validation.count(state) for state in sorted(VALIDATION_STATES)},
        "sigml_records": sigml_records,
        "clip_records": clip_records,
        "image_records": image_records,
        "missing_or_broken_assets": missing_or_broken_assets,
        "render_results": render_results,
        "model_classes": model_classes,
        "lessons_with_avatar": lessons_with_avatar,
        "avatar_model_exists": MODEL_PATH.is_file(),
        "mapped_asset_count": len(mapped_assets),
        "mapped_asset_paths": set(mapped_assets),
        "unmapped_asset_paths": sorted(record["path"] for record in records if record["path"] not in mapped_assets),
        "avatar_directory_file_count": sum(path.is_file() for path in (ROOT / "static" / "avatar").rglob("*")),
    }


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(data: dict) -> str:
    assets = data["records"]
    renderer_passed = sum(record["render_passed"] for record in data["sigml_records"])
    renderer_failed = sum(record["load_status"] == "RENDER_FAILED" for record in data["sigml_records"])
    signer_validated_assets = sum(record["validation_status"] == "VALIDATED" for record in assets)
    sign_validation = data["validation_counts"]
    missing_lessons = [sign.name for sign in data["signs"] if sign.id not in data["linked_sign_ids"]]
    assets_without_lesson = sum(not record["lessons"] for record in assets)
    signs_without_meaning = sum(not sign.meaning for sign in data["signs"])
    signs_without_description = sum(not sign.description for sign in data["signs"])
    signs_without_movement = sum(not getattr(sign, "movement_notes", "") for sign in data["signs"])
    signs_without_usage = sum(not (getattr(sign, "usage_context", "") or sign.usage) for sign in data["signs"])
    signs_without_validation = sum(not getattr(sign, "validation_status", "") for sign in data["signs"])
    unverified_assets = sum(record["validation_status"] == "UNVERIFIED" for record in assets)
    recognition_supported = sum(bool(sign.recognition_label and sign.recognition_label.upper() in data["model_classes"]) for sign in data["signs"])
    summary = [
        ("Sign records", len(data["signs"])),
        ("Lessons", len(data["lessons"])),
        ("Sign-to-lesson links", len(data["links"])),
        ("Sign-specific assets (SiGML + VRM clips + images)", len(assets)),
        ("Sign-specific assets plus shared VRM model", len(assets) + int(data["avatar_model_exists"])),
        ("All files in static/avatar (includes runtime code/docs)", data["avatar_directory_file_count"]),
        ("Sign-specific assets mapped in SQL", f"{len(assets) - len(data['unmapped_asset_paths'])}/{len(assets)}"),
        ("SiGML assets", len(data["sigml_records"])),
        ("CWASA-tested/rendered SiGML", f"{renderer_passed}/{len(data['sigml_records'])}"),
        ("CWASA render failures", renderer_failed),
        ("XML-parseable SiGML", sum(record["file_status"] == "XML_PARSEABLE" for record in data["sigml_records"])),
        ("VRM JSON clips (structure valid; not runtime tested)", f"{sum(record['file_status'] == 'STRUCTURE_VALID' for record in data['clip_records'])}/{len(data['clip_records'])}"),
        ("Readable sign illustrations", f"{sum(record['file_status'] == 'IMAGE_DECODABLE' for record in data['image_records'])}/{len(data['image_records'])}"),
        ("Missing/broken discovered content assets", data["missing_or_broken_assets"]),
        ("Human-validated sign assets", signer_validated_assets),
        ("Sign records linked to at least one lesson", len(data["linked_sign_ids"])),
        ("Sign records without lesson", len(missing_lessons)),
        ("Assets whose sign has no lesson", assets_without_lesson),
        ("SiGML assets without sign records", sum(not record["sign_record"] for record in data["sigml_records"])),
        ("All sign-specific assets without sign records", len(data["orphaned_assets"])),
        ("Asset paths absent from SQL asset mapping", len(data["unmapped_asset_paths"])),
        ("Lessons containing at least one mapped avatar", f"{data['lessons_with_avatar']}/{len(data['lessons'])}"),
        ("Lessons without mapped avatars", max(0, len(data["lessons"]) - data["lessons_with_avatar"])),
        ("Validated sign records", sign_validation.get("VALIDATED", 0)),
        ("Unverified/review-required sign records", sign_validation.get("UNVERIFIED", 0) + sign_validation.get("REVIEW_REQUIRED", 0)),
        ("Human-validated sign records", sign_validation.get("VALIDATED", 0)),
        ("Recognition-supported sign records", recognition_supported),
        ("Recognition-unsupported sign records", len(data["signs"]) - recognition_supported),
        ("Sign records without meaning", signs_without_meaning),
        ("Sign records without description", signs_without_description),
        ("Sign records without movement notes", signs_without_movement),
        ("Sign records without usage context", signs_without_usage),
        ("Sign records without validation status", signs_without_validation),
        ("Image coverage", f"{len(data['image_records'])}/{len(data['signs'])}"),
        ("Video coverage", f"0/{len(data['signs'])}"),
        ("Unverified avatar/content assets", unverified_assets),
        ("Existing DB avatar-asset mapping rows", data["mapped_asset_count"]),
        ("Shared VRM model file present", data["avatar_model_exists"]),
    ]
    lines = [
        "# BridgeSign Avatar and Sign Content Audit",
        "",
        "Generated from the local filesystem, current SQL database, model class files, and the checked-in CWASA validation report. Run `python tools/audit_avatar_content.py` to refresh.",
        "",
        "## Summary",
        "",
        _table(["Measure", "Value"], [[key, value] for key, value in summary]),
        "",
        "## Validation Notes",
        "",
        "- CWASA `ok` means the asset completed the renderer lifecycle in the checked-in browser validation report. It does not establish sign-language correctness.",
        "- The SiGML README labels the dictionary as draft content. No signer review records were found; validated content remains zero unless a reviewer is explicitly recorded in SQL.",
        "- VRM JSON files are checked only for JSON shape, positive duration, and a tracks list. They are not counted as runtime-tested or sign-validated.",
        "- Image files are decoded to confirm file integrity. Several `.png` paths contain JPEG data; their actual format is reported below.",
        "- Recognition availability is based on the current deployed class list, not on avatar files or content names.",
        "",
        "## A-Z Coverage",
        "",
        _table(
            ["Letter", "Model", "Sign record", "Lesson", "SiGML", "CWASA", "Image", "Video", "Meaning", "Movement notes", "Usage", "Content validation"],
            [[
                item["letter"], "YES" if item["recognition"] else "NO", "YES" if item["sign_record"] else "NO",
                "YES" if item["lesson"] else "NO", "YES" if item["avatar"] else "NO", item["avatar_load_status"],
                "YES" if item["image"] else "NO", "NO", "YES" if item["meaning"] else "NO",
                "YES" if item["movement_notes"] else "NO", "YES" if item["usage_context"] else "NO",
                item["validation_status"] or "NO SIGN RECORD",
            ] for item in data["letters"]],
        ),
        "",
        "## SiGML Inventory",
        "",
        _table(
            ["Label", "Path", "XML", "CWASA result", "Sign record", "Lesson(s)", "Recognition", "Asset state", "Sign validation"],
            [[
                record["label"], record["path"], record["file_status"],
                "PASS" if record["render_passed"] else record["load_status"],
                "YES" if record["sign_record"] else "NO", ", ".join(record["lessons"]) or "NONE",
                "YES" if record["recognition"] else "NO", record["asset_status"], record["validation_status"],
            ] for record in data["sigml_records"]],
        ),
        "",
        "## VRM Clip Inventory",
        "",
        _table(
            ["Label", "Path", "Structure", "Runtime test", "Sign record", "Lesson(s)", "Recognition", "Asset state", "Sign validation"],
            [[
                record["label"], record["path"], record["file_status"], record["load_status"],
                "YES" if record["sign_record"] else "NO", ", ".join(record["lessons"]) or "NONE",
                "YES" if record["recognition"] else "NO", record["asset_status"], record["validation_status"],
            ] for record in data["clip_records"]],
        ),
        "",
        "## Image Inventory",
        "",
        _table(
            ["Label", "Path", "Actual format", "Decode", "Sign record", "Lesson(s)", "Sign validation"],
            [[record["label"], record["path"], record["format"], record["file_status"], "YES" if record["sign_record"] else "NO", ", ".join(record["lessons"]) or "NONE", record["validation_status"]] for record in data["image_records"]],
        ),
        "",
        "## Unmapped and Incomplete Content",
        "",
        f"- SiGML files without a sign record: {', '.join(record['label'] for record in data['sigml_records'] if not record['sign_record']) or 'None'}.",
        f"- SiGML files whose sign has no lesson: {', '.join(record['label'] for record in data['sigml_records'] if not record['lessons']) or 'None'}.",
        f"- Sign records without a lesson: {', '.join(missing_lessons) or 'None'}.",
        f"- Sign records without a usable CWASA SiGML asset: {', '.join(data['missing_avatar_signs']) or 'None'}.",
        f"- Sign-specific asset paths missing from the SQL avatar asset map: {len(data['unmapped_asset_paths'])}.",
        f"- Unmapped paths: {', '.join(data['unmapped_asset_paths']) or 'None'}.",
        "- Sign records with missing meaning, movement notes, usage context, and validation notes are identified in the sign-detail table below.",
        "",
        "## Sign Record Inventory",
        "",
        _table(
            ["Sign", "Content type", "Meaning", "Description", "Usage context", "Movement notes", "Asset", "Recognition", "Validation", "Review notes", "Lesson(s)"],
            [[
                sign.name,
                getattr(sign, "content_type", None) or ("LETTER" if len(sign.name) == 1 else "WORD"),
                sign.meaning or "MISSING", sign.description or "MISSING",
                getattr(sign, "usage_context", None) or sign.usage or "MISSING",
                getattr(sign, "movement_notes", None) or "MISSING",
                sign.avatar_animation or "MISSING", sign.recognition_label or "NOT SUPPORTED",
                getattr(sign, "validation_status", None) or "UNVERIFIED",
                getattr(sign, "validation_notes", None) or "MISSING",
                ", ".join(data["sign_lessons"].get(sign.id, [])) or "NONE",
            ] for sign in data["signs"]],
        ),
        "",
        "## Status Vocabulary",
        "",
        f"- Content validation: {', '.join(sorted(VALIDATION_STATES))}.",
        f"- Asset status: {', '.join(sorted(ASSET_STATES))}.",
        "- File integrity, runtime renderability, and human content validation are distinct states.",
        "",
    ]
    return "\n".join(lines)


def _sign_gap_row(data: dict, sign) -> tuple[str, list[str]]:
    assets = [asset for asset in data["asset_rows"] if asset["sign_id"] == sign.id]
    records = [record for record in data["records"] if record["label"] == sign.name.upper()]
    lessons = data["sign_lessons"].get(sign.id, [])
    recognition = bool(sign.recognition_label and sign.recognition_label.upper() in data["model_classes"])
    avatar_demo = any(
        record["type"] == "SIGML"
        and record["file_status"] != "BROKEN"
        and record["load_status"] in {"ASSET_LOADABLE", "XML_PARSEABLE"}
        and record["asset_status"] not in {"MISSING", "BROKEN", "UNSUPPORTED"}
        for record in records
    )
    image_available = any(record["type"] == "IMAGE" and record["file_status"] == "IMAGE_DECODABLE" for record in records)
    video_available = bool(sign.video_url or any(asset["type"] == "VIDEO" for asset in assets))
    lesson_details = [lesson for lesson in data["lesson_details"] if lesson["title"] in lessons]
    assessment = any(
        lesson["exercise_prompt"] and lesson["exercise_options"] and lesson["quiz_question_count"]
        for lesson in lesson_details
    )
    authored_text = any((value or "").strip() for value in (
        sign.meaning, sign.description, sign.usage_context or sign.usage,
        sign.movement_notes, sign.example_sentence,
    ))
    missing = []
    required_fields = (
        ("meaning", sign.meaning),
        ("category", sign.category if sign.category != "unclassified" else ""),
        ("content type", sign.content_type),
        ("difficulty", sign.difficulty if sign.difficulty != "unspecified" else ""),
        ("usage context", sign.usage_context or sign.usage),
        ("movement notes", sign.movement_notes),
        ("example sentence", sign.example_sentence),
    )
    missing.extend(label for label, value in required_fields if not (value or "").strip())
    if not lessons:
        missing.append("lesson mapping")
    if not (avatar_demo or image_available or video_available):
        missing.append("usable lesson demonstration")
    if not missing:
        completeness = "COMPLETE"
    elif authored_text or lessons or avatar_demo or image_available or video_available or recognition:
        completeness = "PARTIAL"
    else:
        completeness = "INCOMPLETE"

    asset_summary = "; ".join(
        f"{asset['type']} {asset['path']} [{asset['status']}/{asset['load_status']}]"
        for asset in assets
    ) or "NONE"
    capabilities = "; ".join((
        f"avatar_demo={'YES' if avatar_demo else 'NO'}",
        f"image={'YES' if image_available else 'NO'}",
        f"video={'YES' if video_available else 'NO'}",
        f"static_A-Z_recognition={'YES' if recognition else 'NO'}",
        f"camera_practice={'YES' if recognition else 'NO'}",
        f"assessment={'YES' if assessment else 'NO'}",
        f"manual_lesson={'YES' if lessons else 'NO'}",
    ))
    movement_display = sign.movement_notes or "MISSING"
    row = [
        sign.id,
        sign.name,
        sign.meaning or "MISSING",
        sign.category or "MISSING",
        sign.content_type or "MISSING",
        sign.difficulty or "MISSING",
        sign.usage_context or sign.usage or "MISSING",
        movement_display,
        sign.description or "MISSING",
        sign.example_sentence or "MISSING",
        sign.recognition_label or "NONE",
        capabilities,
        ", ".join(lessons) or "NONE",
        asset_summary,
        sign.validation_status or "UNVERIFIED",
        completeness,
        ", ".join(missing) or "NONE",
    ]
    return completeness, row


def build_gap_report(data: dict) -> str:
    sign_rows = []
    classifications = {}
    for sign in data["signs"]:
        status, row = _sign_gap_row(data, sign)
        classifications[sign.id] = status
        sign_rows.append(row)

    lesson_rows = []
    for lesson in data["lesson_details"]:
        signs = [sign for sign in data["signs"] if sign.id in lesson["sign_ids"]]
        avatar_signs = [sign.name for sign in signs if _sign_gap_row(data, sign)[1][11].split(";")[0].endswith("YES")]
        image_signs = [sign.name for sign in signs if any(record["type"] == "IMAGE" and record["label"] == sign.name.upper() and record["file_status"] == "IMAGE_DECODABLE" for record in data["records"])]
        video_signs = [sign.name for sign in signs if sign.video_url]
        movement_signs = [sign.name for sign in signs if sign.movement_notes or sign.description]
        example_signs = [sign.name for sign in signs if sign.example_sentence]
        practice_signs = [sign.name for sign in signs if sign.recognition_label and sign.recognition_label.upper() in data["model_classes"]]
        missing = []
        if not lesson["objective"]:
            missing.append("objective")
        if not lesson["steps"]:
            missing.append("teaching sequence")
        if not lesson["sign_ids"]:
            missing.append("mapped signs")
        if not avatar_signs:
            missing.append("avatar demonstration")
        if not image_signs:
            missing.append("image reference")
        if not video_signs:
            missing.append("video reference")
        if len(movement_signs) != len(signs):
            missing.append("written movement/handshape explanation for " + ", ".join(sign.name for sign in signs if sign.name not in movement_signs))
        if len(example_signs) != len(signs):
            missing.append("example sentence for " + ", ".join(sign.name for sign in signs if sign.name not in example_signs))
        if not practice_signs:
            missing.append("camera practice")
        if not lesson["exercise_prompt"] or not lesson["exercise_options"] or not lesson["quiz_question_count"]:
            missing.append("mini challenge or assessment")
        lesson_rows.append([
            lesson["title"],
            lesson["objective"] or "MISSING",
            ", ".join(lesson["sign_names"]) or "NONE",
            " -> ".join(lesson["steps"]) or "MISSING",
            f"{len(avatar_signs)}/{len(signs)} ({', '.join(avatar_signs) or 'none'})",
            f"{len(image_signs)}/{len(signs)} ({', '.join(image_signs) or 'none'})",
            f"{len(video_signs)}/{len(signs)} ({', '.join(video_signs) or 'none'})",
            f"{len(movement_signs)}/{len(signs)} ({', '.join(movement_signs) or 'none'})",
            f"{len(example_signs)}/{len(signs)} ({', '.join(example_signs) or 'none'})",
            f"{len(practice_signs)}/{len(signs)} ({', '.join(practice_signs) or 'none'})",
            "YES" if lesson["exercise_prompt"] and lesson["exercise_options"] and lesson["quiz_question_count"] else "NO",
            f"{len(practice_signs)}/{len(signs)} static A-Z labels",
            "; ".join(missing) or "NONE",
        ])

    missing_lessons = [sign.name for sign in data["signs"] if not data["sign_lessons"].get(sign.id)]
    no_teaching = [
        sign.name for sign in data["signs"]
        if not any((value or "").strip() for value in (
            sign.meaning, sign.description, sign.usage_context or sign.usage,
            sign.movement_notes, sign.example_sentence,
        ))
    ]
    demo_labels = {record["label"] for record in data["sigml_records"] if record["load_status"] in {"ASSET_LOADABLE", "XML_PARSEABLE"} and record["file_status"] != "BROKEN"}
    mapped_pairs = [(link.lesson_id, link.sign_id) for link in data["links"]]
    pair_counts = {}
    for pair in mapped_pairs:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    signs_with_multiple_lessons = [
        sign.name for sign in data["signs"]
        if len(set(data["sign_lessons"].get(sign.id, []))) > 1
    ]
    linked_ids = set(data["linked_sign_ids"])
    recognition_count = sum(bool(sign.recognition_label and sign.recognition_label.upper() in data["model_classes"]) for sign in data["signs"])
    asset_orphans = [asset for asset in data["asset_rows"] if asset["sign_id"] is None]
    summary = [
        ("Sign records", len(data["signs"])),
        ("Lessons / mapped links", f"{len(data['lessons'])} / {len(data['links'])}"),
        ("COMPLETE / PARTIAL / INCOMPLETE", " / ".join(str(sum(status == key for status in classifications.values())) for key in ("COMPLETE", "PARTIAL", "INCOMPLETE"))),
        ("Signs with any mapped avatar asset", len({asset["sign_id"] for asset in data["asset_rows"] if asset["sign_id"] is not None})),
        ("Signs with usable SiGML lesson demo", len(demo_labels)),
        ("Signs with image / video", f"{len(data['image_records'])} / {sum(bool(sign.video_url) for sign in data['signs'])}"),
        ("Signs with static A-Z recognition / camera practice", f"{recognition_count} / {recognition_count}"),
        ("Dynamic gesture recognition", "NOT IMPLEMENTED"),
        ("Signs without lesson", len(missing_lessons)),
        ("Signs without meaning / usage / movement notes / example", f"{sum(not sign.meaning for sign in data['signs'])} / {sum(not (sign.usage_context or sign.usage) for sign in data['signs'])} / {sum(not sign.movement_notes for sign in data['signs'])} / {sum(not sign.example_sentence for sign in data['signs'])}"),
        ("Human-validated signs", data["validation_counts"].get("VALIDATED", 0)),
        ("Sign-specific assets without Sign mapping", len(data["orphaned_assets"])),
        ("Database assets without Sign mapping (shared resources included)", len(asset_orphans)),
        ("Lessons without mapped signs", sum(not lesson["sign_ids"] for lesson in data["lesson_details"])),
        ("Duplicate lesson-sign mappings", sum(count - 1 for count in pair_counts.values())),
        ("Signs mapped to multiple lessons", len(signs_with_multiple_lessons)),
    ]
    unlinked_rows = []
    for sign in data["signs"]:
        if sign.name not in missing_lessons:
            continue
        unlinked_rows.append([
            sign.id, sign.name, sign.category, sign.content_type,
            sign.meaning or "MISSING", sign.usage_context or sign.usage or "MISSING",
            "No lesson assigned; current category is unclassified and meaning/usage are blank. No existing lesson is justified by repository fields. Requires curriculum/content review.",
        ])

    shared_asset_rows = [[asset["type"], asset["path"], asset["status"], asset["load_status"], "Shared resource; no individual Sign mapping expected."] for asset in asset_orphans]
    return "\n".join([
        "# BridgeSign Learning Content Gap Report",
        "",
        "Generated from the current SQL database, local assets, model class list, and checked-in CWASA report by `python tools/audit_avatar_content.py`. This report is an inventory, not linguistic certification.",
        "",
        "## Scope and Criteria",
        "",
        "Content completeness is separate from linguistic validation. `COMPLETE` requires meaning, a classified category, content type, specified difficulty, usage context, movement notes, an example sentence, a lesson mapping, and at least one usable lesson demonstration. `PARTIAL` means some authored text, lesson mapping, recognition, or usable demonstration exists but one or more required fields are absent. `INCOMPLETE` means none of those usable teaching paths is present. Validation status does not alter this classification.",
        "",
        "The local asset catalog labels the language `ASL`. No MSL content or signer review is inferred. Missing language-specific content requires review/content authoring for the language intended by the project.",
        "",
        "## Summary",
        "",
        _table(["Measure", "Current value"], [[key, value] for key, value in summary]),
        "",
        "## Sign Inventory",
        "",
        _table(["ID", "Sign", "Meaning", "Category", "Content type", "Difficulty", "Usage context", "Movement notes", "Existing description", "Example sentence", "Recognition label", "Capability flags", "Lesson mapping", "Avatar asset mapping and states", "Validation status", "Completeness", "Missing components"], sign_rows),
        "",
        "`avatar_demo` means a usable SiGML asset in the current lesson player; a structurally valid VRM clip alone is not counted as a playable lesson demonstration. `camera_practice` is the existing static A-Z classifier only. No dynamic gesture recognition is present.",
        "",
        "## Lesson Coverage",
        "",
        _table(["Lesson", "Objective", "Signs taught", "Teaching sequence", "Avatar demonstrations", "Images", "Videos", "Movement/handshape explanation", "Example sentences", "Practice", "Assessment", "Recognition", "Missing components"], lesson_rows),
        "",
        "## Orphans and Mapping Integrity",
        "",
        f"- Signs without a lesson: {', '.join(missing_lessons) or 'None'}.",
        f"- Lessons without signs: {', '.join(lesson['title'] for lesson in data['lesson_details'] if not lesson['sign_ids']) or 'None'}.",
        f"- Signs without a usable SiGML lesson demonstration: {', '.join(sorted(set(sign.name for sign in data['signs']) - demo_labels)) or 'None'}.",
        f"- Signs with avatar resources but no written teaching content (meaning, description, usage, movement notes, or example): {', '.join(no_teaching) or 'None'}.",
        f"- Sign-specific asset files without a Sign record: {', '.join(record['label'] for record in data['orphaned_assets']) or 'None'}.",
        f"- Duplicate LessonSign pairs: {sum(count - 1 for count in pair_counts.values())}.",
        f"- Signs assigned to multiple lesson titles (review for conflicting curriculum placement): {', '.join(signs_with_multiple_lessons) or 'None'}.",
        "- Database assets with no individual Sign mapping:",
        _table(["Type", "Path", "Asset status", "Load status", "Interpretation"], shared_asset_rows or [["NONE", "NONE", "NONE", "NONE", "No unmapped database assets."]]),
        "",
        "## Unmapped Sign Decisions",
        "",
        "No new lesson mappings were added. Each of the following records has `category=unclassified`, no meaning or usage context, and no lesson association. The repository does not provide enough curriculum evidence to place them in an existing lesson. This is an explicit review decision, not an omitted audit item.",
        "",
        _table(["ID", "Sign", "Current category", "Content type", "Meaning", "Usage", "Decision"], unlinked_rows),
        "",
        "## Recognition and Learning Modes",
        "",
        _table(["Mode", "Current coverage", "Interpretation"], [
            ["Static A-Z recognition", f"{recognition_count} labels", "One-frame classifier supports static letter labels A-Z; not full sign-language recognition."],
            ["Dynamic gesture recognition", "Not implemented", "No temporal/dynamic sign recognition is claimed or changed in this phase."],
            ["Avatar-only demonstration", f"{len(demo_labels - set(data['model_classes']))} signs", "A usable SiGML demonstration exists without a corresponding current static recognition class."],
            ["Manual lesson content", f"{len(linked_ids)} linked sign records; {len(data['signs']) - len(linked_ids)} unlinked", "Lesson mapping and written sign-level content are reported separately; a lesson link alone does not establish linguistic completeness."],
        ]),
        "",
        "## Human Review and Remaining Content Work",
        "",
        f"- Human-validated signs: {data['validation_counts'].get('VALIDATED', 0)}. Technical render success, asset presence, and camera predictions do not constitute signer validation.",
        f"- Missing meaning: {sum(not sign.meaning for sign in data['signs'])}; missing usage context: {sum(not (sign.usage_context or sign.usage) for sign in data['signs'])}; missing movement notes: {sum(not sign.movement_notes for sign in data['signs'])}; missing example sentences: {sum(not sign.example_sentence for sign in data['signs'])}.",
        "- Keep unsupported fields empty until a qualified language reviewer/content author supplies them. For language-specific gaps, use: `Requires human validation/content authoring for the language represented by the source assets (ASL).` Do not convert or label these records as MSL without appropriate source material and review.",
        "- Preserve current validation states. No content is marked validated based on asset rendering, asset presence, or model support.",
        "",
    ])


def main() -> int:
    data = collect_data()
    report = build_report(data)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    GAP_REPORT_PATH.write_text(build_gap_report(data), encoding="utf-8")
    print(f"Report: {OUTPUT_PATH}")
    print(f"Sign records: {len(data['signs'])}; lessons: {len(data['lessons'])}")
    print(f"SiGML: {sum(record['render_passed'] for record in data['sigml_records'])}/{len(data['sigml_records'])} CWASA rendered")
    print(f"VRM clips: {len(data['clip_records'])}; images: {len(data['image_records'])}")
    print(f"Orphan SiGML: {sum(not record['sign_record'] for record in data['sigml_records'])}; validated signs: {data['validation_counts'].get('VALIDATED', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
