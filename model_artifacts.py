"""Helpers for validating on-disk model artifacts before deserialization."""

from pathlib import Path


_GIT_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer(path):
    """Return whether a file contains a Git LFS pointer instead of model data."""
    try:
        with Path(path).open("rb") as artifact:
            return artifact.readline(128).rstrip(b"\r\n") == _GIT_LFS_POINTER_HEADER
    except OSError:
        return False


def remove_unresolved_git_lfs_pointer(path):
    """Delete stale Git LFS pointer stubs so newer valid models can be used."""
    target = Path(path)
    if not target.exists() or not is_git_lfs_pointer(target):
        return False
    try:
        target.unlink()
        return True
    except OSError:
        return False


def remove_stale_git_lfs_pointers(directory):
    """Remove all Git LFS pointer stubs from a directory of model artifacts."""
    if not directory:
        return []
    root = Path(directory)
    if not root.exists():
        return []

    removed = []
    for artifact in sorted(root.iterdir()):
        if artifact.is_file() and remove_unresolved_git_lfs_pointer(artifact):
            removed.append(str(artifact))
    return removed

MANIFEST_FILENAME = "model_manifest.json"


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 hex digest of a file, or None if it cannot be read."""
    import hashlib

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def load_manifest(directory):
    """Load models/model_manifest.json as {filename: entry}; empty when absent or invalid."""
    import json

    if not directory:
        return {}
    path = Path(directory) / MANIFEST_FILENAME
    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            data = json.load(manifest_file)
    except (OSError, ValueError):
        return {}
    artifacts = data.get("artifacts") if isinstance(data, dict) else None
    if not isinstance(artifacts, dict):
        return {}
    return {name: entry for name, entry in artifacts.items() if isinstance(entry, dict)}


def check_against_manifest(path, labels=None, n_features=None, manifest=None):
    """
    Compare a loaded artifact with its manifest entry.

    Returns (errors, warnings, entry). Errors are mismatches that make the
    model unusable for the live feature format (labels, feature count);
    warnings are provenance differences such as a changed checksum.
    """
    name = Path(path).name
    if manifest is None:
        manifest = load_manifest(Path(path).parent)
    entry = manifest.get(name)
    if entry is None:
        return [], [f"{name} has no entry in {MANIFEST_FILENAME}; provenance is unrecorded."], None

    errors, warnings = [], []
    expected_labels = entry.get("labels")
    if labels is not None and expected_labels is not None and list(labels) != list(expected_labels):
        errors.append(f"{name} labels do not match {MANIFEST_FILENAME}.")
    expected_features = entry.get("n_features")
    if n_features is not None and expected_features is not None and int(n_features) != int(expected_features):
        errors.append(
            f"{name} expects {n_features} features; {MANIFEST_FILENAME} records {expected_features}."
        )
    expected_sha = entry.get("sha256")
    if expected_sha and sha256_file(path) != expected_sha:
        warnings.append(f"{name} differs from the SHA-256 recorded in {MANIFEST_FILENAME}.")
    return errors, warnings, entry
