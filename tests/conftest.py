"""Shared pytest setup for the pymrsf test suite.

Makes `import pymrsf` work whether or not the package is installed (src layout),
and ensures importing pymrsf never tries to load a multi-GB local model (default
provider is overridden to a no-op API provider). Individual tests opt into the
local provider via set_provider().
"""
import os
import sys

# Repo root is the parent of tests/; the package lives under src/ (src layout).
# Prefer the installed package, but fall back to src/ so the suite also runs
# directly from a source checkout without `pip install -e .`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

# Pick a provider that does NOT load a GGUF at import time. Must be set before
# `import pymrsf` happens anywhere in the session.
os.environ.setdefault("PYMRSF_PROVIDER", "openai")

# Path to a local GGUF for the (optional) local-provider tests. Override with
# PYMRSF_TEST_MODEL; defaults to the model the author tested against.
DEFAULT_TEST_MODEL = r"C:\Users\pokam\Downloads\Databirck\models\mistral-7b-v0.1.Q4_K_M.gguf"


def local_model_path():
    return os.getenv("PYMRSF_TEST_MODEL", DEFAULT_TEST_MODEL)
