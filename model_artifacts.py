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