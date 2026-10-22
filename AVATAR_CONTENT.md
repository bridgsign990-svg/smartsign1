# Avatar and Sign Content

## Architecture

Sign-language content is stored in the existing `signs` table. `lesson_signs` links one or more signs to a lesson in display order. `avatar_assets` maps a sign, or a shared avatar resource, to a repository-relative path and records file type, language, version, asset status, load status, validation state, and notes.

These states have different meanings:

- `asset_status`: `AVAILABLE`, `MISSING`, `BROKEN`, `DRAFT`, `TEST_ONLY`, or `UNSUPPORTED` describes whether a local resource exists and is intended for use.
- `load_status`: records a narrow technical check such as `ASSET_LOADABLE`, `XML_PARSEABLE`, `IMAGE_DECODABLE`, `STRUCTURE_VALID`, `RENDER_FAILED`, or `NOT_TESTED`.
- `validation_status`: `UNVERIFIED`, `REVIEW_REQUIRED`, `VALIDATED`, `REJECTED`, or `DEPRECATED` describes human review of the sign content. A successful renderer check does not set this to `VALIDATED`.
- `recognition_label`: is independent. `recognition_available` is true only if that label is present in the currently configured model class list.

The Learn UI uses the mapped SiGML path from SQL through the existing CWASA bridge. It does not infer a path from a sign name. VRM motion clips are cataloged, but the lesson CWASA player does not claim they are playback-ready.

## Current Assets

The repository contains 50 SiGML drafts, 36 VRM JSON clips, six sign illustrations, and one shared VRM model. The checked-in CWASA browser report records successful playback of all 50 SiGML labels. The clips pass only a JSON-structure check; they have not been runtime-tested by the catalog audit. All these assets remain linguistically unverified. Six files stored with a `.png` extension contain JPEG bytes; the audit reports the detected format without renaming them.

The recognition model class list is A-Z. Camera evaluation is available only for signs explicitly mapped to one of those labels. An avatar asset never grants recognition support by itself.

## Adding Content

1. Use the established sign label as the unique `Sign.name`. Set `content_type` to the appropriate value such as `LETTER`, `NUMBER`, `WORD`, `PHRASE`, `SENTENCE`, `GESTURE`, or `SITUATION`.
2. Populate meaning, movement notes, usage context, example sentence, and common mistakes only from a reliable project source or an identified human reviewer. Otherwise leave them empty and keep the record `UNVERIFIED` or `REVIEW_REQUIRED`.
3. Add an asset only when the actual file exists. The current filesystem sync maps matching SiGML, VRM clip, and image filenames to the same normalized sign label. It creates an empty unverified sign record for a file without a record; it does not synthesize descriptions, URLs, animations, or recognition labels.
4. For a nonstandard path or multiple assets of one type, create an `AvatarAsset` row with the exact repository path, type, format, language, version, and honest status. Keep `Sign.avatar_animation` only as a Phase 2 compatibility field; the learning API reads normalized asset mappings.
5. Add the sign to a lesson with a `LessonSign` association. A lesson may reference multiple signs; content types are not restricted to the alphabet.
6. Run `python tools/audit_avatar_content.py` and review `AVATAR_CONTENT_AUDIT.md` before relying on the mapping.

The startup sync is idempotent. It preserves reviewer states, marks previously cataloged files `MISSING` when they disappear, and never deletes asset files or learner progress.

## Human Review

No content review/admin UI is currently provided. Learner endpoints are read-only for shared content. A trusted content-maintenance workflow must record `reviewed_by`, `reviewed_at`, `validation_notes`, and `validation_status` on the sign, and independently set `AvatarAsset.validation_status` for the reviewed media. Do not set `VALIDATED` merely because XML parses or CWASA renders.

When an avatar is missing or broken, the lesson shows the written content and an explicit unavailable state. When recognition is unsupported, it explains that camera evaluation is unavailable. Missing descriptions and movement notes are shown as unavailable rather than inferred.

## Audit

Run from the project root with the configured project environment:

```powershell
python tools/audit_avatar_content.py
```

The report compares filesystem assets, SQL mappings, lesson links, model classes, image decoding, SiGML XML parsing, and the checked-in CWASA render report. `ASSET_LOADABLE` and renderer `PASS` are technical outcomes only; content validation remains a separate human state.
