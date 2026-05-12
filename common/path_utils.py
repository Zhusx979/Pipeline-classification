from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_file(path: str | Path, *, description: str) -> Path:
    resolved = resolve_project_path(path)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def ensure_directory(path: str | Path, *, description: str) -> Path:
    resolved = resolve_project_path(path)
    if resolved is None or not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def sanitize_filename(text: str) -> str:
    sanitized = re.sub(r"[^\w.+-]+", "_", text.strip(), flags=re.UNICODE)
    return sanitized.strip("._") or "experiment"


def build_timestamped_output_dir(base_root: str | Path, experiment_name: str, date_tag: str) -> Path:
    base_dir = resolve_project_path(base_root)
    if base_dir is None:
        raise ValueError("base_root must not be None")
    dated_root = base_dir.parent / f"{base_dir.name}_{date_tag}"
    return dated_root / sanitize_filename(experiment_name)
