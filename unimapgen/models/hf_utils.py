import os


def resolve_hf_snapshot_path(path: str) -> str:
    path = str(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    snapshots_dir = os.path.join(path, "snapshots")
    refs_main = os.path.join(path, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            ref = f.read().strip()
        cand = os.path.join(snapshots_dir, ref)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    if os.path.isdir(snapshots_dir):
        snaps = sorted(
            [
                os.path.join(snapshots_dir, x)
                for x in os.listdir(snapshots_dir)
                if os.path.isfile(os.path.join(snapshots_dir, x, "config.json"))
            ]
        )
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(f"Unable to resolve HuggingFace snapshot under: {path}")
