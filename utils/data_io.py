# utils/data_io.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from .schema import detect_mission, to_standard

def read_csv_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)

def ingest_file(path: Path) -> pd.DataFrame:
    raw = read_csv_any(path)
    mission = detect_mission(raw)
    std = to_standard(raw, mission)
    std["__filename__"] = path.name
    return std

def build_catalog(data_dir: Path, extra_files: list[Path] | None = None, mode: str = "builtin") -> pd.DataFrame:
    files = []
    if mode in ("builtin","append"):
        for fname in ["cumulative_2025.10.04_11.34.12.csv","TOI_2025.10.04_11.34.21.csv","k2pandc_2025.10.04_11.34.25.csv"]:
            fp = data_dir / fname
            if fp.exists(): files.append(fp)
    if mode in ("append","replace") and extra_files:
        files.extend(extra_files)
    frames = [ingest_file(p) for p in files]
    if not frames:
        return pd.DataFrame()
    cat = pd.concat(frames, ignore_index=True)
    cat = cat.drop_duplicates(subset=["mission","source_id","label"], keep="last")
    return cat

def summary_counts(cat: pd.DataFrame) -> pd.DataFrame:
    if cat.empty: return pd.DataFrame()
    return cat.groupby(["mission","label"], dropna=False).size().reset_index(name="count")
