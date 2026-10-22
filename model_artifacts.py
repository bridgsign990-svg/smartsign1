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