#!/usr/bin/env python3
"""Readiness includes both the loaded model and the exact released gallery."""
import json
import os
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=4) as response:
    health = json.load(response)
if not health.get("model_loaded") or not health.get("storage", {}).get("reachable"):
    raise SystemExit("Model/storage not ready")
if health["storage"].get("gallery_points") != int(os.environ.get("EXPECTED_GALLERY_POINTS", "750")):
    raise SystemExit("Released gallery not ready")
