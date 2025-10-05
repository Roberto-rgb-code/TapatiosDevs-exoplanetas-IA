# models/pipeline.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# ========================
# 🔬 Features estándar
# ========================
FEATURES_NUM = [
    "radius_re",
    "orbital_period",
    "duration_hours",
    "depth_ppm",
    "insol",
    "teff",
    "star_rad_rs",
]
FEATURES_CAT = ["mission"]

def _prepare_dataframe(cat: pd.DataFrame) -> pd.DataFrame:
    """Limpia NaNs de label y rellena numéricos con la mediana por columna."""
    df = cat.dropna(subset=["label"]).copy()

    # Asegurar existencia de columnas num/cat aunque falten en alguna misión
    for c in FEATURES_NUM:
        if c not in df.columns:
            df[c] = np.nan
    for c in FEATURES_CAT:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    # Relleno robusto para numéricos
    for c in FEATURES_NUM:
        med = pd.to_numeric(df[c], errors="coerce").median()
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)

    # Quitar filas con mission vacía
    df["mission"] = df["mission"].astype(str).replace({"": "UNKNOWN"})
    return df

# ========================
# 🧠 Clasificador
# ========================
def make_classifier():
    base = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        learning_rate=0.1,
        n_estimators=220,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=0,
    )
    # Nota: en sklearn recientes el parámetro es "estimator" (no base_estimator)
    return CalibratedClassifierCV(estimator=base, cv=3, method="isotonic")

# ========================
# 🔁 Entrenamiento completo
# Devuelve (pipe, metrics, label_encoder)
# ========================
def train_model(cat: pd.DataFrame, test_subset="TESS"):
    df = _prepare_dataframe(cat)

    # Etiquetas → enteros (XGBoost las prefiere así)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str))

    X = df[FEATURES_NUM + FEATURES_CAT].copy()

    # Split inteligente: si el subset de test deja vacío el train, usar stratified split
    use_subset = (test_subset in df["mission"].unique())
    if use_subset:
        mask = (df["mission"] == test_subset)
        if (~mask).sum() < 1 or mask.sum() < 1:
            use_subset = False

    if use_subset:
        X_train, X_test = X[~mask], X[mask]
        y_train, y_test = y[~mask], y[mask]
    else:
        # 25% test estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

    # Preprocesador
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ],
        remainder="drop",
        n_jobs=None,
    )

    pipe = Pipeline([
        ("prep", preproc),
        ("clf", make_classifier()),
    ])

    pipe.fit(X_train, y_train)

    # Evaluación
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    labels_idx = np.unique(y_test)
    labels_names = le.inverse_transform(labels_idx)

    # AUC macro OvR (si algo falla, lo omitimos)
    auc_macro_ovr = None
    try:
        # one-vs-rest sobre columnas en el mismo orden que classes_
        y_test_bin = np.zeros_like(y_prob)
        for i, cls in enumerate(pipe.classes_):
            y_test_bin[:, i] = (y_test == cls).astype(int)
        auc_macro_ovr = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        pass

    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        output_dict=True
    )

    metrics = {
        "report": report,
        "confusion_matrix": confusion_matrix(
            le.inverse_transform(y_test),
            le.inverse_transform(y_pred),
            labels=labels_names
        ).tolist(),
        "labels": list(labels_names),
        "auc_macro_ovr": auc_macro_ovr,
    }

    return pipe, metrics, le
