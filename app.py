"""
ProwriteAI — Root Entrypoint for Cloud Deployment (Render, Streamlit Cloud, Docker)

Delegates execution to ProwriteAI_SmartText_project_v8/app.py while ensuring
all relative paths (checkpoints, artifacts, outputs) are correctly located.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve project root and inner package directory
PROJECT_ROOT = Path(__file__).resolve().parent
INNER_APP_DIR = PROJECT_ROOT / "ProwriteAI_SmartText_project_v8"

# Add inner directory to sys.path so modules and local resources resolve seamlessly
if str(INNER_APP_DIR) not in sys.path:
    sys.path.insert(0, str(INNER_APP_DIR))

# Set default environment variables if not already provided
os.environ.setdefault("PROWRITE_CHECKPOINT_DIR", str(INNER_APP_DIR / "prowrite_outputs" / "checkpoints"))
os.environ.setdefault("PROWRITE_ARTIFACT_DIR", str(INNER_APP_DIR / "prowrite_outputs" / "artifacts"))
os.environ.setdefault("PROWRITE_OUTPUT_DIR", str(INNER_APP_DIR / "prowrite_outputs"))
os.environ.setdefault("PROWRITE_DATA_DIR", str(INNER_APP_DIR / "data"))

# Load and execute the primary application
inner_app_path = INNER_APP_DIR / "app.py"
if not inner_app_path.exists():
    raise FileNotFoundError(f"Primary app.py not found at: {inner_app_path}")

with open(inner_app_path, "r", encoding="utf-8") as f:
    code = f.read()

# Execute within the context of the inner directory
__file__ = str(inner_app_path)
exec(compile(code, str(inner_app_path), "exec"), globals())
