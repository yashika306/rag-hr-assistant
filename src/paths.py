"""
paths.py
--------
Central place for filesystem paths used across the project, so every
module resolves the same locations no matter which directory a script
is *run from* (a common gotcha with `adk web`, `adk api_server`, and
scripts invoked from different working directories).
"""

import os

# Absolute path to the project root (the folder that contains src/, data/, etc.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "hr_policy.pdf")

CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
CHROMA_COLLECTION_NAME = "hr_policy_docs"
