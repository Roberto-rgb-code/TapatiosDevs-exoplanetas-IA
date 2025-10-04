# utils/schema.py
from __future__ import annotations
import pandas as pd
import numpy as np

# 1) Columnas "firma" para detectar misión
FINGERPRINTS = {
    "KEPLER": ["koi_pdisposition", "koi_period", "koi_depth"],
    "TESS":   ["tfopwg_disp", "pl_orbper", "pl_trandep"],
    "K2":     ["disposition", "pl_orbper", "pl_rade"],
}

# 2) Mapeo a nombres estándar (por bloque)
MAP_STANDARD = {
    # Etiqueta/label
    "label": {"KEPLER": "koi_pdisposition", "TESS": "tfopwg_disp", "K2": "disposition"},
    # Tránsito
    "orbital_period": {"KEPLER":"koi_period", "TESS":"pl_orbper", "K2":"pl_orbper"},
    "duration_hours": {"KEPLER":"koi_duration","TESS":"pl_trandurh","K2":None},
    "depth_ppm":      {"KEPLER":"koi_depth",   "TESS":"pl_trandep", "K2":None},
    "impact":         {"KEPLER":"koi_impact",  "TESS":None,         "K2":None},
    "epoch":          {"KEPLER":"koi_time0bk", "TESS":"pl_tranmid", "K2":None},
    "snr":            {"KEPLER":"koi_model_snr","TESS":None,        "K2":None},
    # Planeta
    "radius_re":      {"KEPLER":"koi_prad", "TESS":"pl_rade", "K2":"pl_rade"},
    "temp_eq":        {"KEPLER":"koi_teq",  "TESS":"pl_eqt",  "K2":"pl_eqt"},
    "insol":          {"KEPLER":"koi_insol","TESS":"pl_insol","K2":"pl_insol"},
    "ecc":            {"KEPLER":None,       "TESS":None,      "K2":"pl_orbeccen"},
    # Estrella
    "teff":           {"KEPLER":"koi_steff","TESS":"st_teff", "K2":"st_teff"},
    "logg":           {"KEPLER":"koi_slogg","TESS":"st_logg", "K2":"st_logg"},
    "star_rad_rs":    {"KEPLER":"koi_srad", "TESS":"st_rad",  "K2":"st_rad"},
    "mag":            {"KEPLER":"koi_kepmag","TESS":"st_tmag","K2":"sy_vmag"},
    "dist_pc":        {"KEPLER":None,       "TESS":"st_dist", "K2":"sy_dist"},
    # Posición
    "ra":             {"KEPLER":"ra", "TESS":"ra", "K2":"ra"},
    "dec":            {"KEPLER":"dec","TESS":"dec","K2":"dec"},
    # Flags de falsos positivos (Kepler)
    "fp_nt":          {"KEPLER":"koi_fpflag_nt","TESS":None,"K2":None},
    "fp_ss":          {"KEPLER":"koi_fpflag_ss","TESS":None,"K2":None},
    "fp_co":          {"KEPLER":"koi_fpflag_co","TESS":None,"K2":None},
    "fp_ec":          {"KEPLER":"koi_fpflag_ec","TESS":None,"K2":None},
}

# 3) Tipos destino por columna estándar
TARGET_DTYPES = {
    "label": "category",
    "orbital_period": "float64",
    "duration_hours": "float64",
    "depth_ppm": "float64",
    "impact": "float64",
    "epoch": "float64",
    "snr": "float64",
    "radius_re": "float64",
    "temp_eq": "float64",
    "insol": "float64",
    "ecc": "float64",
    "teff": "float64",
    "logg": "float64",
    "star_rad_rs": "float64",
    "mag": "float64",
    "dist_pc": "float64",
    "ra": "float64",
    "dec": "float64",
    "fp_nt": "boolean",
    "fp_ss": "boolean",
    "fp_co": "boolean",
    "fp_ec": "boolean",
    "mission": "category",
    "source_id": "string",
}

# ---------- Utilidades ----------

def map_label(mission: str, s: pd.Series) -> pd.Series:
    """Mapea etiquetas por misión a las clases armonizadas."""
    s = s.astype(str).str.upper()
    if mission == "TESS":
        mapping = {
            "CP":"CANDIDATE","PC":"CANDIDATE","APC":"CANDIDATE",
            "FP":"FALSE POSITIVE",
            "KP":"CONFIRMED","PK":"CONFIRMED"
        }
        return s.map(lambda x: mapping.get(x, x))
    # KEPLER ya trae texto completo; K2 suele venir en texto estándar
    return s

def detect_mission(df: pd.DataFrame) -> str:
    """Detecta la misión por columnas presentes."""
    cols = set(df.columns)
    for m, fp in FINGERPRINTS.items():
        if all(c in cols for c in fp):
            return m
    if "koi_pdisposition" in cols: return "KEPLER"
    if "tfopwg_disp" in cols: return "TESS"
    if "disposition" in cols: return "K2"
    return "UNKNOWN"

def _empty_series(dtype: str, index) -> pd.Series:
    """Serie vacía con dtype correcto (evita errores de NAType)."""
    if dtype == "boolean":
        return pd.Series(pd.array([pd.NA] * len(index), dtype="boolean"), index=index)
    if dtype == "category":
        return pd.Series(pd.Categorical([None] * len(index)), index=index)
    if dtype == "string":
        return pd.Series([None] * len(index), dtype="string", index=index)
    # Numéricos: usar np.nan
    return pd.Series([np.nan] * len(index), dtype=dtype, index=index)

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fuerza tipos destino; crea columnas faltantes con valores válidos."""
    for c, t in TARGET_DTYPES.items():
        if c in df.columns:
            if t == "boolean":
                df[c] = (
                    df[c]
                    .replace({"True": 1, "False": 0, "Y": 1, "N": 0, "1": 1, "0": 0})
                    .astype("float64")
                )
                df[c] = (df[c] == 1.0).astype("boolean")
            elif t == "category":
                df[c] = df[c].astype("string").astype("category")
            elif t == "string":
                df[c] = df[c].astype("string")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(t)
        else:
            df[c] = _empty_series(t, df.index)
    return df

def to_standard(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    """
    Proyecta un DataFrame de KOI/TESS/K2 al esquema estándar y tipa columnas.
    Añade 'mission' y una 'source_id' (mejor esfuerzo).
    """
    out = pd.DataFrame(index=df.index)
    out["mission"] = mission

    # Etiqueta
    lbl_col = MAP_STANDARD["label"].get(mission)
    if lbl_col and lbl_col in df.columns:
        out["label"] = map_label(mission, df[lbl_col])

    # Features por mapeo
    for std_name, per_m in MAP_STANDARD.items():
        if std_name == "label":
            continue
        src = per_m.get(mission)
        if src and src in df.columns:
            out[std_name] = df[src]

    # ID de origen (mejor esfuerzo)
    cand_ids = [c for c in ["kepoi_name","kepler_name","kepid","toi","pl_name"] if c in df.columns]
    out["source_id"] = df[cand_ids[0]].astype(str) if cand_ids else pd.Series(np.nan, index=df.index)

    # Tipado final
    out = coerce_dtypes(out)
    return out
