import json
import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

import config


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(80), primary_key=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    phrases: Mapped[list["EmergencyPhrase"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class EmergencyPhrase(Base):
    __tablename__ = "emergency_phrases"

    username: Mapped[str] = mapped_column(ForeignKey("users.username", ondelete="CASCADE"), primary_key=True)
    phrase_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    phrase: Mapped[str] = mapped_column(Text, nullable=False)
    user: Mapped[User] = relationship(back_populates="phrases")


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    level: Mapped[str] = mapped_column(String(40), default="beginner", nullable=False)


class Module(Base):
    __tablename__ = "modules"
    __table_args__ = (UniqueConstraint("course_id", "order_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_number: Mapped[int] = mapped_column(Integer, nullable=False)


class Lesson(Base):
    __tablename__ = "lessons"
    __table_args__ = (UniqueConstraint("module_id", "order_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    difficulty: Mapped[str] = mapped_column(String(40), default="beginner", nullable=False)
    passing_score: Mapped[int] = mapped_column(Integer, default=70, nullable=False)
    order_number: Mapped[int] = mapped_column(Integer, nullable=False)
    objective: Mapped[str] = mapped_column(Text, default="", nullable=False)
    introduction: Mapped[str] = mapped_column(Text, default="", nullable=False)
    steps_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    exercise_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    exercise_options_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    exercise_answer: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    exercise_explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)


class Sign(Base):
    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    meaning: Mapped[str] = mapped_column(Text, default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    usage: Mapped[str] = mapped_column(Text, default="", nullable=False)
    avatar_animation: Mapped[str | None] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(80), default="everyday", nullable=False)
    difficulty: Mapped[str] = mapped_column(String(40), default="beginner", nullable=False)
    recognition_label: Mapped[str | None] = mapped_column(String(80))
    video_url: Mapped[str | None] = mapped_column(String(500))
    image_url: Mapped[str | None] = mapped_column(String(500))
    common_mistakes_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    usage_context: Mapped[str] = mapped_column(Text, default="", nullable=False)
    movement_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    example_sentence: Mapped[str] = mapped_column(Text, default="", nullable=False)
    content_type: Mapped[str] = mapped_column(String(24), default="WORD", nullable=False)
    validation_status: Mapped[str] = mapped_column(String(24), default="UNVERIFIED", nullable=False)
    validation_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(120))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    avatar_assets: Mapped[list["AvatarAsset"]] = relationship(back_populates="sign", cascade="all, delete-orphan")


class AvatarAsset(Base):
    __tablename__ = "avatar_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sign_id: Mapped[int | None] = mapped_column(ForeignKey("signs.id", ondelete="CASCADE"))
    asset_type: Mapped[str] = mapped_column(String(24), nullable=False)
    path: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    format: Mapped[str] = mapped_column(String(24), nullable=False)
    language: Mapped[str] = mapped_column(String(24), default="UNSPECIFIED", nullable=False)
    version: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    asset_status: Mapped[str] = mapped_column(String(24), default="AVAILABLE", nullable=False)
    validation_status: Mapped[str] = mapped_column(String(24), default="UNVERIFIED", nullable=False)
    load_status: Mapped[str] = mapped_column(String(32), default="NOT_TESTED", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    sign: Mapped[Sign | None] = relationship(back_populates="avatar_assets")


class LessonSign(Base):
    __tablename__ = "lesson_signs"

    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), primary_key=True)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True)
    order_number: Mapped[int] = mapped_column(Integer, nullable=False)


class UserLessonProgress(Base):
    __tablename__ = "user_lesson_progress"
    __table_args__ = (UniqueConstraint("username", "lesson_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="locked", nullable=False)
    progress_percentage: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    best_score: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_section: Mapped[str] = mapped_column(String(40), default="introduction", nullable=False)
    last_activity: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active_quiz_session: Mapped[str | None] = mapped_column(String(36))
    exercise_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    total_questions: Mapped[int] = mapped_column(Integer, nullable=False)
    percentage: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"
    __table_args__ = (UniqueConstraint("lesson_id", "order_number"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False)
    question_type: Mapped[str] = mapped_column(String(40), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    sign_id: Mapped[int | None] = mapped_column(ForeignKey("signs.id", ondelete="SET NULL"))
    options_json: Mapped[str] = mapped_column(Text, nullable=False)
    answer_index: Mapped[int] = mapped_column(Integer, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_number: Mapped[int] = mapped_column(Integer, nullable=False)


class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False)
    questions_json: Mapped[str] = mapped_column(Text, nullable=False)
    answers_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class PracticeAttempt(Base):
    __tablename__ = "practice_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    lesson_id: Mapped[int] = mapped_column(ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id", ondelete="CASCADE"), nullable=False)
    target_label: Mapped[str] = mapped_column(String(80), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    feedback: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserSignMastery(Base):
    __tablename__ = "user_sign_mastery"

    username: Mapped[str] = mapped_column(ForeignKey("users.username", ondelete="CASCADE"), primary_key=True)
    sign_id: Mapped[int] = mapped_column(ForeignKey("signs.id", ondelete="CASCADE"), primary_key=True)
    practice_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mastery_score: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    last_practiced: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


_engine = None
_Session = None


def _database_url():
    url = os.environ.get("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return "sqlite:///" + os.path.join(config.DATA_DIR, "bridgesign.sqlite3").replace("\\", "/")


def initialize_database(database_url=None, legacy_data_dir=None):
    global _engine, _Session
    url = database_url or _database_url()
    if _engine is not None:
        _engine.dispose()
    options = {"pool_pre_ping": True}
    if url.startswith("sqlite:"):
        options["connect_args"] = {"check_same_thread": False}
    _engine = create_engine(url, **options)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    _upgrade_existing_schema()
    migrate_legacy_json(legacy_data_dir or config.DATA_DIR)


def _upgrade_existing_schema():
    additions = {
        "lessons": {
            "objective": "TEXT NOT NULL DEFAULT ''",
            "introduction": "TEXT NOT NULL DEFAULT ''",
            "steps_json": "TEXT NOT NULL DEFAULT '[]'",
            "exercise_prompt": "TEXT NOT NULL DEFAULT ''",
            "exercise_options_json": "TEXT NOT NULL DEFAULT '[]'",
            "exercise_answer": "INTEGER NOT NULL DEFAULT 0",
            "exercise_explanation": "TEXT NOT NULL DEFAULT ''",
        },
        "signs": {
            "category": "VARCHAR(80) NOT NULL DEFAULT 'everyday'",
            "difficulty": "VARCHAR(40) NOT NULL DEFAULT 'beginner'",
            "recognition_label": "VARCHAR(80)",
            "video_url": "VARCHAR(500)",
            "image_url": "VARCHAR(500)",
            "common_mistakes_json": "TEXT NOT NULL DEFAULT '[]'",
            "usage_context": "TEXT NOT NULL DEFAULT ''",
            "movement_notes": "TEXT NOT NULL DEFAULT ''",
            "example_sentence": "TEXT NOT NULL DEFAULT ''",
            "content_type": "VARCHAR(24) NOT NULL DEFAULT 'WORD'",
            "validation_status": "VARCHAR(24) NOT NULL DEFAULT 'UNVERIFIED'",
            "validation_notes": "TEXT NOT NULL DEFAULT ''",
            "reviewed_by": "VARCHAR(120)",
            "reviewed_at": "TIMESTAMP",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "user_lesson_progress": {
            "current_section": "VARCHAR(40) NOT NULL DEFAULT 'introduction'",
            "last_activity": "TIMESTAMP",
            "active_quiz_session": "VARCHAR(36)",
            "exercise_completed": "BOOLEAN NOT NULL DEFAULT FALSE",
        },
        "quiz_attempts": {
            "attempt_number": "INTEGER NOT NULL DEFAULT 1",
            "started_at": "TIMESTAMP",
        },
    }
    inspector = inspect(_engine)
    with _engine.begin() as connection:
        for table_name, columns in additions.items():
            if table_name not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, definition in columns.items():
                if column_name not in existing:
                    connection.exec_driver_sql(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {definition}'
                    )

        refreshed_inspector = inspect(_engine)
        if "signs" in refreshed_inspector.get_table_names():
            sign_columns = {column["name"] for column in refreshed_inspector.get_columns("signs")}
            if {"usage", "usage_context"}.issubset(sign_columns):
                connection.exec_driver_sql(
                    "UPDATE signs SET usage_context = usage WHERE usage_context = '' AND usage <> ''"
                )
            if {"meaning", "description", "validation_status", "validation_notes"}.issubset(sign_columns):
                connection.exec_driver_sql(
                    "UPDATE signs SET validation_status = 'REVIEW_REQUIRED', "
                    "validation_notes = 'Existing content has not been reviewed by a fluent sign-language reviewer.' "
                    "WHERE validation_status = 'UNVERIFIED' AND (meaning <> '' OR description <> '')"
                )
            if {"created_at", "updated_at"}.issubset(sign_columns):
                connection.exec_driver_sql(
                    "UPDATE signs SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
                )
                connection.exec_driver_sql(
                    "UPDATE signs SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL"
                )


def _session():
    if _Session is None:
        initialize_database()
    return _Session()


def migrate_legacy_json(data_dir):
    users_path = os.path.join(data_dir, "users.json")
    if os.path.isfile(users_path):
        try:
            with open(users_path, encoding="utf-8") as file:
                users = json.load(file)
            if isinstance(users, dict):
                upsert_users(users)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[Database] Could not migrate {users_path}: {error}")

    if not os.path.isdir(data_dir):
        return
    for filename in os.listdir(data_dir):
        if not filename.startswith("phrases_") or not filename.endswith(".json"):
            continue
        username = filename[len("phrases_"):-len(".json")]
        try:
            with open(os.path.join(data_dir, filename), encoding="utf-8") as file:
                phrases = json.load(file)
            if isinstance(phrases, dict) and phrases:
                save_user_phrases(username, phrases)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[Database] Could not migrate {filename}: {error}")


def load_users():
    with _session() as session:
        return {user.username: {"password": user.password_hash} for user in session.scalars(select(User))}


def get_user(username):
    with _session() as session:
        user = session.get(User, username)
        return {"password": user.password_hash} if user else None


def upsert_users(users):
    with _session() as session, session.begin():
        for username, data in users.items():
            password_hash = data.get("password") if isinstance(data, dict) else None
            if not username or not password_hash:
                continue
            user = session.get(User, username)
            if user is None:
                session.add(User(username=username, password_hash=password_hash))


def create_user(username, password_hash):
    try:
        with _session() as session, session.begin():
            session.add(User(username=username, password_hash=password_hash))
        return True
    except IntegrityError:
        return False


def load_user_phrases(username):
    with _session() as session:
        return {
            phrase.phrase_id: phrase.phrase
            for phrase in session.scalars(select(EmergencyPhrase).where(EmergencyPhrase.username == username))
        }


def save_user_phrases(username, phrases):
    with _session() as session, session.begin():
        if session.get(User, username) is None:
            return
        session.query(EmergencyPhrase).filter_by(username=username).delete()
        session.add_all(
            EmergencyPhrase(username=username, phrase_id=str(phrase_id), phrase=str(phrase))
            for phrase_id, phrase in phrases.items()
        )