# BridgeSign Avatar and Sign Content Audit

Generated from the local filesystem, current SQL database, model class files, and the checked-in CWASA validation report. Run `python tools/audit_avatar_content.py` to refresh.

## Summary

| Measure | Value |
| --- | --- |
| Sign records | 51 |
| Lessons | 9 |
| Sign-to-lesson links | 34 |
| Sign-specific assets (SiGML + VRM clips + images) | 92 |
| Sign-specific assets plus shared VRM model | 93 |
| All files in static/avatar (includes runtime code/docs) | 94 |
| Sign-specific assets mapped in SQL | 92/92 |
| SiGML assets | 50 |
| CWASA-tested/rendered SiGML | 50/50 |
| CWASA render failures | 0 |
| XML-parseable SiGML | 50 |
| VRM JSON clips (structure valid; not runtime tested) | 36/36 |
| Readable sign illustrations | 6/6 |
| Missing/broken discovered content assets | 0 |
| Human-validated sign assets | 0 |
| Sign records linked to at least one lesson | 34 |
| Sign records without lesson | 17 |
| Assets whose sign has no lesson | 18 |
| SiGML assets without sign records | 0 |
| All sign-specific assets without sign records | 0 |
| Asset paths absent from SQL asset mapping | 0 |
| Lessons containing at least one mapped avatar | 9/9 |
| Lessons without mapped avatars | 0 |
| Validated sign records | 0 |
| Unverified/review-required sign records | 51 |
| Human-validated sign records | 0 |
| Recognition-supported sign records | 26 |
| Recognition-unsupported sign records | 25 |
| Sign records without meaning | 40 |
| Sign records without description | 40 |
| Sign records without movement notes | 51 |
| Sign records without usage context | 40 |
| Sign records without validation status | 0 |
| Image coverage | 6/51 |
| Video coverage | 0/51 |
| Unverified avatar/content assets | 64 |
| Existing DB avatar-asset mapping rows | 93 |
| Shared VRM model file present | True |

## Validation Notes

- CWASA `ok` means the asset completed the renderer lifecycle in the checked-in browser validation report. It does not establish sign-language correctness.
- The SiGML README labels the dictionary as draft content. No signer review records were found; validated content remains zero unless a reviewer is explicitly recorded in SQL.
- VRM JSON files are checked only for JSON shape, positive duration, and a tracks list. They are not counted as runtime-tested or sign-validated.
- Image files are decoded to confirm file integrity. Several `.png` paths contain JPEG data; their actual format is reported below.
- Recognition availability is based on the current deployed class list, not on avatar files or content names.

## A-Z Coverage

| Letter | Model | Sign record | Lesson | SiGML | CWASA | Image | Video | Meaning | Movement notes | Usage | Content validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | YES | YES | YES | YES | ASSET_LOADABLE | YES | NO | YES | NO | YES | REVIEW_REQUIRED |
| B | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | YES | NO | YES | REVIEW_REQUIRED |
| C | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | YES | NO | YES | REVIEW_REQUIRED |
| D | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| E | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| F | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| G | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| H | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| I | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| J | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| K | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| L | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| M | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| N | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| O | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| P | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| Q | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| R | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| S | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| T | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| U | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| V | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| W | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| X | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| Y | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |
| Z | YES | YES | YES | YES | ASSET_LOADABLE | NO | NO | NO | NO | NO | UNVERIFIED |

## SiGML Inventory

| Label | Path | XML | CWASA result | Sign record | Lesson(s) | Recognition | Asset state | Sign validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | /static/avatar/sigml/asl/A.sigml | XML_PARSEABLE | PASS | YES | Letters A to C | YES | DRAFT | REVIEW_REQUIRED |
| AMBULANCE | /static/avatar/sigml/asl/AMBULANCE.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| B | /static/avatar/sigml/asl/B.sigml | XML_PARSEABLE | PASS | YES | Letters A to C | YES | DRAFT | REVIEW_REQUIRED |
| C | /static/avatar/sigml/asl/C.sigml | XML_PARSEABLE | PASS | YES | Letters A to C | YES | DRAFT | REVIEW_REQUIRED |
| D | /static/avatar/sigml/asl/D.sigml | XML_PARSEABLE | PASS | YES | Letters D to H | YES | DRAFT | UNVERIFIED |
| DANGER | /static/avatar/sigml/asl/DANGER.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| DOCTOR | /static/avatar/sigml/asl/DOCTOR.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| E | /static/avatar/sigml/asl/E.sigml | XML_PARSEABLE | PASS | YES | Letters D to H | YES | DRAFT | UNVERIFIED |
| EMERGENCY | /static/avatar/sigml/asl/EMERGENCY.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| F | /static/avatar/sigml/asl/F.sigml | XML_PARSEABLE | PASS | YES | Letters D to H | YES | DRAFT | UNVERIFIED |
| FIRE | /static/avatar/sigml/asl/FIRE.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| FOOD | /static/avatar/sigml/asl/FOOD.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| G | /static/avatar/sigml/asl/G.sigml | XML_PARSEABLE | PASS | YES | Letters D to H | YES | DRAFT | UNVERIFIED |
| H | /static/avatar/sigml/asl/H.sigml | XML_PARSEABLE | PASS | YES | Letters D to H | YES | DRAFT | UNVERIFIED |
| HELLO | /static/avatar/sigml/asl/HELLO.sigml | XML_PARSEABLE | PASS | YES | Greetings | NO | DRAFT | REVIEW_REQUIRED |
| HELP | /static/avatar/sigml/asl/HELP.sigml | XML_PARSEABLE | PASS | YES | Ask for Help | NO | DRAFT | REVIEW_REQUIRED |
| HURT | /static/avatar/sigml/asl/HURT.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| I | /static/avatar/sigml/asl/I.sigml | XML_PARSEABLE | PASS | YES | Letters I to M | YES | DRAFT | UNVERIFIED |
| J | /static/avatar/sigml/asl/J.sigml | XML_PARSEABLE | PASS | YES | Letters I to M | YES | DRAFT | UNVERIFIED |
| K | /static/avatar/sigml/asl/K.sigml | XML_PARSEABLE | PASS | YES | Letters I to M | YES | DRAFT | UNVERIFIED |
| L | /static/avatar/sigml/asl/L.sigml | XML_PARSEABLE | PASS | YES | Letters I to M | YES | DRAFT | UNVERIFIED |
| M | /static/avatar/sigml/asl/M.sigml | XML_PARSEABLE | PASS | YES | Letters I to M | YES | DRAFT | UNVERIFIED |
| MUG | /static/avatar/sigml/asl/MUG.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| N | /static/avatar/sigml/asl/N.sigml | XML_PARSEABLE | PASS | YES | Letters N to R | YES | DRAFT | UNVERIFIED |
| NO | /static/avatar/sigml/asl/NO.sigml | XML_PARSEABLE | PASS | YES | Yes and No | NO | DRAFT | REVIEW_REQUIRED |
| O | /static/avatar/sigml/asl/O.sigml | XML_PARSEABLE | PASS | YES | Letters N to R | YES | DRAFT | UNVERIFIED |
| P | /static/avatar/sigml/asl/P.sigml | XML_PARSEABLE | PASS | YES | Letters N to R | YES | DRAFT | UNVERIFIED |
| PAIN | /static/avatar/sigml/asl/PAIN.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| PLEASE | /static/avatar/sigml/asl/PLEASE.sigml | XML_PARSEABLE | PASS | YES | Greetings | NO | DRAFT | REVIEW_REQUIRED |
| POLICE | /static/avatar/sigml/asl/POLICE.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| Q | /static/avatar/sigml/asl/Q.sigml | XML_PARSEABLE | PASS | YES | Letters N to R | YES | DRAFT | UNVERIFIED |
| R | /static/avatar/sigml/asl/R.sigml | XML_PARSEABLE | PASS | YES | Letters N to R | YES | DRAFT | UNVERIFIED |
| S | /static/avatar/sigml/asl/S.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| SICK | /static/avatar/sigml/asl/SICK.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| SORRY | /static/avatar/sigml/asl/SORRY.sigml | XML_PARSEABLE | PASS | YES | Thanks and Apologies | NO | DRAFT | REVIEW_REQUIRED |
| STOP | /static/avatar/sigml/asl/STOP.sigml | XML_PARSEABLE | PASS | YES | Ask for Help | NO | DRAFT | REVIEW_REQUIRED |
| T | /static/avatar/sigml/asl/T.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| TAKE | /static/avatar/sigml/asl/TAKE.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| THANK_YOU | /static/avatar/sigml/asl/THANK_YOU.sigml | XML_PARSEABLE | PASS | YES | Thanks and Apologies | NO | DRAFT | REVIEW_REQUIRED |
| TOILET | /static/avatar/sigml/asl/TOILET.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| U | /static/avatar/sigml/asl/U.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| V | /static/avatar/sigml/asl/V.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| W | /static/avatar/sigml/asl/W.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| WATER | /static/avatar/sigml/asl/WATER.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| WHAT | /static/avatar/sigml/asl/WHAT.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| WHERE | /static/avatar/sigml/asl/WHERE.sigml | XML_PARSEABLE | PASS | YES | NONE | NO | DRAFT | UNVERIFIED |
| X | /static/avatar/sigml/asl/X.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| Y | /static/avatar/sigml/asl/Y.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |
| YES | /static/avatar/sigml/asl/YES.sigml | XML_PARSEABLE | PASS | YES | Yes and No | NO | DRAFT | REVIEW_REQUIRED |
| Z | /static/avatar/sigml/asl/Z.sigml | XML_PARSEABLE | PASS | YES | Letters S to Z | YES | DRAFT | UNVERIFIED |

## VRM Clip Inventory

| Label | Path | Structure | Runtime test | Sign record | Lesson(s) | Recognition | Asset state | Sign validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | /static/avatar/animations/A.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters A to C | YES | AVAILABLE | REVIEW_REQUIRED |
| B | /static/avatar/animations/B.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters A to C | YES | AVAILABLE | REVIEW_REQUIRED |
| C | /static/avatar/animations/C.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters A to C | YES | AVAILABLE | REVIEW_REQUIRED |
| D | /static/avatar/animations/D.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters D to H | YES | AVAILABLE | UNVERIFIED |
| E | /static/avatar/animations/E.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters D to H | YES | AVAILABLE | UNVERIFIED |
| F | /static/avatar/animations/F.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters D to H | YES | AVAILABLE | UNVERIFIED |
| G | /static/avatar/animations/G.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters D to H | YES | AVAILABLE | UNVERIFIED |
| GOODBYE | /static/avatar/animations/GOODBYE.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | NONE | NO | AVAILABLE | UNVERIFIED |
| H | /static/avatar/animations/H.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters D to H | YES | AVAILABLE | UNVERIFIED |
| HELLO | /static/avatar/animations/HELLO.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Greetings | NO | AVAILABLE | REVIEW_REQUIRED |
| HELP | /static/avatar/animations/HELP.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Ask for Help | NO | AVAILABLE | REVIEW_REQUIRED |
| I | /static/avatar/animations/I.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters I to M | YES | AVAILABLE | UNVERIFIED |
| J | /static/avatar/animations/J.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters I to M | YES | AVAILABLE | UNVERIFIED |
| K | /static/avatar/animations/K.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters I to M | YES | AVAILABLE | UNVERIFIED |
| L | /static/avatar/animations/L.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters I to M | YES | AVAILABLE | UNVERIFIED |
| M | /static/avatar/animations/M.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters I to M | YES | AVAILABLE | UNVERIFIED |
| N | /static/avatar/animations/N.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters N to R | YES | AVAILABLE | UNVERIFIED |
| NO | /static/avatar/animations/NO.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Yes and No | NO | AVAILABLE | REVIEW_REQUIRED |
| O | /static/avatar/animations/O.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters N to R | YES | AVAILABLE | UNVERIFIED |
| P | /static/avatar/animations/P.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters N to R | YES | AVAILABLE | UNVERIFIED |
| PLEASE | /static/avatar/animations/PLEASE.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Greetings | NO | AVAILABLE | REVIEW_REQUIRED |
| Q | /static/avatar/animations/Q.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters N to R | YES | AVAILABLE | UNVERIFIED |
| R | /static/avatar/animations/R.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters N to R | YES | AVAILABLE | UNVERIFIED |
| S | /static/avatar/animations/S.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| SORRY | /static/avatar/animations/SORRY.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Thanks and Apologies | NO | AVAILABLE | REVIEW_REQUIRED |
| STOP | /static/avatar/animations/STOP.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Ask for Help | NO | AVAILABLE | REVIEW_REQUIRED |
| T | /static/avatar/animations/T.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| THANK_YOU | /static/avatar/animations/THANK_YOU.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Thanks and Apologies | NO | AVAILABLE | REVIEW_REQUIRED |
| U | /static/avatar/animations/U.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| V | /static/avatar/animations/V.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| W | /static/avatar/animations/W.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| WATER | /static/avatar/animations/WATER.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | NONE | NO | AVAILABLE | UNVERIFIED |
| X | /static/avatar/animations/X.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| Y | /static/avatar/animations/Y.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |
| YES | /static/avatar/animations/YES.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Yes and No | NO | AVAILABLE | REVIEW_REQUIRED |
| Z | /static/avatar/animations/Z.json | STRUCTURE_VALID | NOT_RUNTIME_TESTED | YES | Letters S to Z | YES | AVAILABLE | UNVERIFIED |

## Image Inventory

| Label | Path | Actual format | Decode | Sign record | Lesson(s) | Sign validation |
| --- | --- | --- | --- | --- | --- | --- |
| A | /static/signs/sign_a.png | JPEG | IMAGE_DECODABLE | YES | Letters A to C | REVIEW_REQUIRED |
| HELLO | /static/signs/sign_hello.png | JPEG | IMAGE_DECODABLE | YES | Greetings | REVIEW_REQUIRED |
| HELP | /static/signs/sign_help.png | JPEG | IMAGE_DECODABLE | YES | Ask for Help | REVIEW_REQUIRED |
| NO | /static/signs/sign_no.png | JPEG | IMAGE_DECODABLE | YES | Yes and No | REVIEW_REQUIRED |
| THANK_YOU | /static/signs/sign_thankyou.png | JPEG | IMAGE_DECODABLE | YES | Thanks and Apologies | REVIEW_REQUIRED |
| YES | /static/signs/sign_yes.png | JPEG | IMAGE_DECODABLE | YES | Yes and No | REVIEW_REQUIRED |

## Unmapped and Incomplete Content

- SiGML files without a sign record: None.
- SiGML files whose sign has no lesson: AMBULANCE, DANGER, DOCTOR, EMERGENCY, FIRE, FOOD, HURT, MUG, PAIN, POLICE, SICK, TAKE, TOILET, WATER, WHAT, WHERE.
- Sign records without a lesson: AMBULANCE, DANGER, DOCTOR, EMERGENCY, FIRE, FOOD, GOODBYE, HURT, MUG, PAIN, POLICE, SICK, TAKE, TOILET, WATER, WHAT, WHERE.
- Sign records without a usable CWASA SiGML asset: GOODBYE.
- Sign-specific asset paths missing from the SQL avatar asset map: 0.
- Unmapped paths: None.
- Sign records with missing meaning, movement notes, usage context, and validation notes are identified in the sign-detail table below.

## Sign Record Inventory

| Sign | Content type | Meaning | Description | Usage context | Movement notes | Asset | Recognition | Validation | Review notes | Lesson(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | LETTER | Fingerspelled letter A | Close the fingers into a fist with the thumb resting alongside. | Use when fingerspelling a name or word containing A. | MISSING | sigml:A | A | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Letters A to C |
| AMBULANCE | WORD | MISSING | MISSING | MISSING | MISSING | sigml:AMBULANCE | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| B | LETTER | Fingerspelled letter B | Hold the fingers straight together with the thumb folded across the palm. | Use when fingerspelling a name or word containing B. | MISSING | sigml:B | B | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Letters A to C |
| C | LETTER | Fingerspelled letter C | Curve the fingers and thumb into a C-shaped hand. | Use when fingerspelling a name or word containing C. | MISSING | sigml:C | C | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Letters A to C |
| D | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:D | D | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters D to H |
| DANGER | WORD | MISSING | MISSING | MISSING | MISSING | sigml:DANGER | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| DOCTOR | WORD | MISSING | MISSING | MISSING | MISSING | sigml:DOCTOR | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| E | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:E | E | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters D to H |
| EMERGENCY | WORD | MISSING | MISSING | MISSING | MISSING | sigml:EMERGENCY | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| F | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:F | F | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters D to H |
| FIRE | WORD | MISSING | MISSING | MISSING | MISSING | sigml:FIRE | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| FOOD | WORD | MISSING | MISSING | MISSING | MISSING | sigml:FOOD | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| G | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:G | G | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters D to H |
| GOODBYE | WORD | MISSING | MISSING | MISSING | MISSING | MISSING | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| H | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:H | H | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters D to H |
| HELLO | WORD | A greeting | Open hand near the temple, then move it outward in a greeting motion. | Use to greet someone or open an interaction. | MISSING | sigml:HELLO | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Greetings |
| HELP | WORD | A request for assistance | Place a thumb-up hand on the other palm and lift both hands slightly. | Use to ask someone for assistance. | MISSING | sigml:HELP | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Ask for Help |
| HURT | WORD | MISSING | MISSING | MISSING | MISSING | sigml:HURT | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| I | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:I | I | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters I to M |
| J | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:J | J | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters I to M |
| K | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:K | K | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters I to M |
| L | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:L | L | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters I to M |
| M | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:M | M | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters I to M |
| MUG | WORD | MISSING | MISSING | MISSING | MISSING | sigml:MUG | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| N | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:N | N | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters N to R |
| NO | WORD | A negative response | Bring the extended fingers toward the thumb in a short closing motion. | Use to decline or answer negatively. | MISSING | sigml:NO | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Yes and No |
| O | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:O | O | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters N to R |
| P | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:P | P | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters N to R |
| PAIN | WORD | MISSING | MISSING | MISSING | MISSING | sigml:PAIN | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| PLEASE | WORD | A courteous request | Place a flat hand on the chest and move it in a circular motion. | Use to make a request more polite. | MISSING | sigml:PLEASE | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Greetings |
| POLICE | WORD | MISSING | MISSING | MISSING | MISSING | sigml:POLICE | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| Q | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:Q | Q | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters N to R |
| R | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:R | R | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters N to R |
| S | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:S | S | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| SICK | WORD | MISSING | MISSING | MISSING | MISSING | sigml:SICK | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| SORRY | WORD | An apology | Make a fist and rub it in a small circle on the chest. | Use to apologize or express regret. | MISSING | sigml:SORRY | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Thanks and Apologies |
| STOP | WORD | A request to halt | Bring a flat hand down onto the other open palm. | Use to ask someone to stop an action. | MISSING | sigml:STOP | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Ask for Help |
| T | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:T | T | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| TAKE | WORD | MISSING | MISSING | MISSING | MISSING | sigml:TAKE | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| THANK_YOU | WORD | An expression of gratitude | Start with a flat hand near the chin and move it forward. | Use to express thanks. | MISSING | sigml:THANK_YOU | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Thanks and Apologies |
| TOILET | WORD | MISSING | MISSING | MISSING | MISSING | sigml:TOILET | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| U | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:U | U | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| V | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:V | V | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| W | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:W | W | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| WATER | WORD | MISSING | MISSING | MISSING | MISSING | sigml:WATER | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| WHAT | WORD | MISSING | MISSING | MISSING | MISSING | sigml:WHAT | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| WHERE | WORD | MISSING | MISSING | MISSING | MISSING | sigml:WHERE | NOT SUPPORTED | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | NONE |
| X | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:X | X | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| Y | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:Y | Y | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |
| YES | WORD | An affirmative response | Make a fist and move it in a small nodding motion. | Use to agree or answer affirmatively. | MISSING | sigml:YES | NOT SUPPORTED | REVIEW_REQUIRED | Existing content has not been reviewed by a fluent sign-language reviewer. | Yes and No |
| Z | LETTER | MISSING | MISSING | MISSING | MISSING | sigml:Z | Z | UNVERIFIED | An avatar or image file exists; reviewed sign content has not been provided. | Letters S to Z |

## Status Vocabulary

- Content validation: DEPRECATED, REJECTED, REVIEW_REQUIRED, UNVERIFIED, VALIDATED.
- Asset status: AVAILABLE, BROKEN, DRAFT, MISSING, TEST_ONLY, UNSUPPORTED.
- File integrity, runtime renderability, and human content validation are distinct states.
