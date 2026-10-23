# External MS-ASL Dataset Guide

## Dataset and purpose

MS-ASL is an **American Sign Language (ASL)** video dataset. BridgeSign keeps this
external dataset separate so researchers can validate the dynamic video/landmark
pipeline with an existing ASL resource. It is not evidence of MSL coverage or accuracy.

> This dataset must not be treated as Malawian Sign Language training data.

Long-term separation:

```text
External ASL datasets
        ↓
Pipeline/model experimentation

Actual MSL data
        ↓
BridgeSign MSL model
```

## Obtain metadata and review terms

Obtain the official MS-ASL metadata archive or extracted JSON files from the
dataset's official publication/repository. The importer recognizes:

- `MSASL_train.json`
- `MSASL_val.json`
- `MSASL_test.json`
- `MSASL_classes.json`
- `MSASL_synonym.json`

The archive may be supplied directly, or extracted to a directory. The importer
copies the recognized metadata into `data/external_datasets/msasl/metadata/` and
records the input source. A ZIP is read by filename; it is not unpacked blindly.

Review the official MS-ASL license/terms and restrictions on each linked source
video before downloading or using any data. Source videos may be hosted by third
parties and can be deleted, private, restricted, or unavailable. The importer
uses only the metadata URL as provided. It does not bypass access controls, search
for substitutes, or scrape unrelated websites. If license information is absent
from the supplied archive, the report says so and directs the user to check the
official terms.

## Import a small metadata sample

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl --metadata "path\to\MS-ASL.zip" --limit 100
```

The default cap is 100 selected records; specify `--limit 10`, `--limit 50`, or
`--limit 500` as appropriate. The selected sample is deterministic and retains
the official train/val/test allocation. Importing metadata alone does not make
network requests.

Preview without downloads:

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl --metadata "path\to\metadata-directory" --limit 10 --dry-run
```

The full metadata manifest is written to `metadata/msasl_metadata.csv`; the
selected records and their download/processing statuses are in
`manifests/import_manifest.csv`.

## Download a controlled sample

Only invoke `--download` when the source terms permit it:

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl --download
```

Downloads go into `data/external_datasets/msasl/raw/`. Existing local files are
reused. Transient failures receive bounded retries and are recorded as `FAILED`;
HTTP 401/403/404/410 responses are recorded as `UNAVAILABLE`. One failed URL does
not stop processing the rest of the selected sample. The importer requests only
direct resources from the metadata URL; HTML pages are not treated as video files.
No command downloads the entire dataset implicitly.

## Clip extraction and preprocessing

After eligible videos are available locally:

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl --process
```

When valid start/end times are provided, OpenCV extracts that segment to
`data/external_datasets/msasl/raw/clips/`. Missing times cause the full source video
to be copied as a technical fallback and explicitly noted; malformed or unusable
segments are recorded as failures. Processing reuses BridgeSign's existing
MediaPipe Holistic landmark extraction, normalization, and temporal-window
implementation. Processed `.npz` windows are stored under
`data/external_datasets/msasl/processed/{train,val,test}/`.

The current MediaPipe Tasks setup needs the Holistic task bundle. The path is
configured in `config/msl_dataset.json` as `holistic_model_path` (default:
`models/holistic_landmarker.task`). Obtain that model asset through official
MediaPipe distribution and configure a local path before processing. No ASL model
is trained by this workflow.

Processed arrays include normalized finite model features, the preserved raw
landmark array (where missing detections remain NaN), a landmark-presence mask,
feature-group names, label and signer metadata, original split, video ID, source
URL, and segment timestamps. ASL label strings remain ASL labels.

## Splits, validation, and reports

The official source split is retained; the importer does not silently reassign
signers. Inspect signer overlap:

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl --split
```

Validate manifests, videos, dimensions, finite model values, and missing landmarks,
then generate reports:

```powershell
.\.venv\Scripts\python.exe -m tools.msasl_dataset.import_msasl --validate --report
```

Outputs:

- `manifests/sequences.csv`
- `manifests/split_manifest.json`
- `reports/MSASL_IMPORT_REPORT.md`
- `reports/MSASL_QUALITY_REPORT.md`
- `reports/dataset_summary.json`
- `reports/quality_summary.json`
- `review/msasl_review_candidates.json`

Review warnings and failures before experiments. The `MSASLDataset` class in
`tools.msasl_dataset.import_msasl` loads only samples explicitly marked `MS-ASL`
and `American Sign Language (ASL)`.

## Supported CLI options

```text
--metadata PATH             Official ZIP, metadata directory, or JSON file
--limit N                   Maximum records selected (default 100)
--dry-run                   Preview selected records; never downloads
--download                  Download selected metadata URLs
--process                   Extract clips and process local downloaded videos
--split                     Summarize official splits and signer overlap
--validate                  Run quality checks and produce review candidates
--report                    Generate import and quality reports
--force                     Replace existing downloads/reprocess sequences
--timeout SECONDS           Per-request timeout (default 30)
--retries N                 Retries after the initial request (default 2)
--expected-feature-count N  Optional strict processed feature width
```

`--help` prints the full command help. Operations can be combined, for example
`--download --process --validate --report`, but downloading remains an explicit
choice.

## Language boundary

MS-ASL is useful only as an **external ASL** pipeline/experimentation dataset.
Its labels and samples must not be copied into `data/msl_dataset/` or treated as
Malawian Sign Language. BridgeSign MSL training and evaluation require real,
appropriately reviewed MSL data collected for that purpose.
