from pathlib import Path


def create_unique_folder(base_path):
    base = Path(base_path)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    idx = 1
    while True:
        candidate = Path(f"{base_path}_{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        idx += 1
