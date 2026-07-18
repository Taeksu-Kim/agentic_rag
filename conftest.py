"""Repo root on sys.path so `agent.*`, `evaluation.*`, `retriever.*` import."""

import pathlib
import sys

_root = str(pathlib.Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
