"""Isolated utilities for importing external MS-ASL (American Sign Language) data."""

__all__ = ["MSASLDataset"]


def __getattr__(name: str):
    if name == "MSASLDataset":
        from .import_msasl import MSASLDataset

        return MSASLDataset
    raise AttributeError(name)
