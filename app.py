# app.py
from __future__ import annotations


# --- Fix PYTHONPATH for Streamlit Cloud ---
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ----

import os, io, json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from plotly.graph_objs import Figure
from joblib import dump
from dotenv import load_dotenv  # para leer .env

from utils.data_io import build_catalog, summary_counts
from models.pipeline import train_model, FEATURES_NUM, FEATURES_CAT
from chatbot.grok_agent import GrokAgent

# === Cargar variables de entorno (.env) ===
load_dotenv()
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4-fast-reasoning")

APP_TITLE = "A World Away: Hunting for Exoplanets with AI"
TEAM_NAME = "TapatiosDevs"

DATA_DIR = Path("data")
ASSETS_DIR = Path("assets")
LOGOS_DIR = ASSETS_DIR / "logos"

st.set_page_config(page_title="Exoplanet AI • TapatiosDevs", page_icon="🌌", layout="wide")

# ---------- Paleta (fondos azul claro)
PAPER_BG = "#2E5AA7"   # fondo exterior de los charts
PLOT_BG  = "#3B6BC4"   # fondo del área de ploteo
TABLE_BG = "#2E5AA7"   # fondo de tablas

# Colores de alto contraste para clases (evita azul oscuro)
COLOR_MAP = {
    "CONFIRMED": "#FFD166",       # dorado claro
    "CANDIDATE": "#06D6A0",       # turquesa
    "FALSE POSITIVE": "#EF476F",  # coral
    "FA": "#A78BFA",              # lavanda
    # variantes de texto posibles
    "Confirmed": "#FFD166",
    "Candidate": "#06D6A0",
    "False Positive": "#EF476F",
}

def style_plot(fig: Figure) -> Figure:
    """Aplica fondos azul claro y márgenes a cualquier figura plotly."""
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

# --- CSS general (mantiene tu style.css y añade tema azul para tablas/containers)
if (ASSETS_DIR / "style.css").exists():
    with open(ASSETS_DIR / "style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
      .stApp {{ background: #0E1A30; }} /* fondo global (oscuro) */
      .stMarkdown, .block-container {{ background: transparent; }}
      div[data-testid="stMetricValue"] {{ color: #fff; }}
      /* DataFrame */
      div[data-testid="stDataFrame"] .st-ag-root-wrapper, 
      div[data-testid="stDataFrame"] .st-ag-root,
      div[data-testid="stDataFrame"] .ag-root-wrapper,
      div[data-testid="stDataFrame"] .ag-root {{
        background-color: {TABLE_BG} !important;
        border-radius: 10px;
      }}
      div[data-testid="stDataFrame"] .ag-header {{
        background-color: {PAPER_BG} !important;
        color: #fff !important;
      }}
      div[data-testid="stDataFrame"] .ag-row, 
      div[data-testid="stDataFrame"] .ag-cell {{
        background-color: {TABLE_BG} !important;
        color: #fff !important;
        border-color: rgba(255,255,255,0.1) !important;
      }}
      .stButton > button {{
        background: #4B82F0;
        color: white;
        border: 0;
        border-radius: 8px;
      }}
      .badge {{
        display:inline-block;padding:.25rem .6rem;border-radius:.5rem;
        background:#4B82F0;color:#fff;font-weight:600;font-size:.85rem;
      }}
      h1,h2,h3,h4,h5 {{ color: #EAF2FF; }}
      .small {{ color:#cfe2ff }}
      hr {{ border: none; border-top: 1px solid rgba(255,255,255,0.15); }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Logos embebidos (sin upload)
def logos_row():
    cols = st.columns(4)
    logos = {
        "NASA": LOGOS_DIR / "NASA Logo.png",
        "Space Apps": LOGOS_DIR / "NASA Space Challenge.png",
        "CUGDL": LOGOS_DIR / "CUGDL LOGO OFICIAL.png",
        "UDG": LOGOS_DIR / "Escudo Universidad de Guadalajara logo white.png",
    }
    for i, (label, path) in enumerate(logos.items()):
        with cols[i]:
            if path.exists():
                st.image(str(path), caption=label, use_container_width=True)
            else:
                st.warning(f"⚠️ Falta el logo de {label}")

# --- Header
st.markdown(
    f"""
<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;">
  <span class="badge">NASA Space Apps 2025</span>
  <h1 style="margin:0;">{APP_TITLE}</h1>
</div>
<div class="small">Equipo: <b>{TEAM_NAME}</b> · Modelo LLM: <b>{GROK_MODEL}</b></div>
<hr/>
""",
    unsafe_allow_html=True,
)
logos_row()

# --- Sidebar: datos (sin selector de misión)
st.sidebar.header("⚙️ Datos")
mode = st.sidebar.selectbox(
    "Fuente de datos",
    ["builtin", "append", "replace"],
    index=0,
    help="builtin: embebidos • append: embebidos + tus archivos • replace: solo tus archivos",
)

extra_files = st.sidebar.file_uploader("Subir CSV adicionales", type=["csv"], accept_multiple_files=True)

uploaded_paths = []
if extra_files:
    for uf in extra_files:
        p = Path(f".cache_upload_{uf.name}")
        with open(p, "wb") as w:
            w.write(uf.read())
        uploaded_paths.append(p)

# --- Construir catálogo y dejar SOLO TESS
cat = build_catalog(DATA_DIR, uploaded_paths, mode=mode)
if cat.empty:
    st.error("No hay datos. Coloca los CSV embebidos en /data o sube archivos.")
    st.stop()

if "mission" not in cat.columns:
    st.error("El catálogo no contiene columna 'mission'.")
    st.stop()

cat = cat[cat["mission"].astype(str).str.upper() == "TESS"].copy()
if cat.empty:
    st.error("No se encontraron filas de la misión TESS en el catálogo cargado.")
    st.stop()

st.success(f"Catálogo activo (solo TESS): {len(cat):,} filas")

# --- Resumen (solo TESS, pero mostramos por clase)
st.subheader("📊 Distribución por clase (TESS)")
cnt = summary_counts(cat)  # devuelve mission/label/count; aquí mission será TESS
st.dataframe(cnt, use_container_width=True)

fig_bar = px.bar(
    cnt, x="label", y="count", color="label", text_auto=True,
    color_discrete_map=COLOR_MAP, title="Conteo por clase en TESS"
)
style_plot(fig_bar)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("---")

# --- Exploración 2D
st.subheader("🔎 Exploración rápida (2D)")
c1, c2 = st.columns(2)
with c1:
    x_axis = st.selectbox("Eje X", ["radius_re", "orbital_period", "depth_ppm", "teff", "insol"], index=0)
with c2:
    y_axis = st.selectbox("Eje Y", ["orbital_period", "duration_hours", "snr", "mag", "star_rad_rs"], index=1)

fig_sc = px.scatter(
    cat, x=x_axis, y=y_axis, color="label",
    opacity=0.88, hover_data=["source_id"], color_discrete_map=COLOR_MAP,
    title="Distribución 2D (TESS)"
)
style_plot(fig_sc)
st.plotly_chart(fig_sc, use_container_width=True)
st.markdown("---")

# --- Exploración 3D
st.subheader("🧭 Exploración 3D (TESS)")
candidates = ["radius_re", "orbital_period", "duration_hours", "depth_ppm",
              "insol", "teff", "star_rad_rs", "snr", "mag"]
num_opts = [c for c in candidates if c in cat.columns]

if len(num_opts) < 3:
    st.info("Se requieren al menos 3 variables numéricas (por ejemplo: radius_re, orbital_period, depth_ppm).")
else:
    c31, c32, c33, c34 = st.columns([1,1,1,1])
    with c31:
        x3 = st.selectbox("Eje X (3D)", num_opts, index=0, key="x3_sel")
    with c32:
        y3 = st.selectbox("Eje Y (3D)", num_opts, index=min(1, len(num_opts)-1), key="y3_sel")
    with c33:
        z3 = st.selectbox("Eje Z (3D)", num_opts, index=min(2, len(num_opts)-1), key="z3_sel")
    with c34:
        size_col = st.selectbox("Tamaño (opcional)", ["(fijo)"] + num_opts, index=0, key="size3_sel")

    df3 = cat[[x3, y3, z3, "label", "source_id"]].dropna().copy()

    # Submuestreo para rendimiento
    max_pts = 5000
    if len(df3) > max_pts:
        df3 = df3.sample(max_pts, random_state=42)

    use_log = st.checkbox("Usar escala log en X/Y/Z", value=False, key="log3_sel")

    kw = {}
    if size_col != "(fijo)" and size_col in cat.columns:
        df3[size_col] = cat.loc[df3.index, size_col]
        kw["size"] = size_col
        kw["size_max"] = 18

    fig3 = px.scatter_3d(
        df3, x=x3, y=y3, z=z3, color="label",
        hover_data={"source_id": True, x3: True, y3: True, z3: True, "label": True},
        color_discrete_map=COLOR_MAP, **kw
    )
    # fondos 3D
    fig3.update_layout(paper_bgcolor=PAPER_BG, font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig3.update_scenes(
        xaxis=dict(backgroundcolor=PLOT_BG, gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.35)"),
        yaxis=dict(backgroundcolor=PLOT_BG, gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.35)"),
        zaxis=dict(backgroundcolor=PLOT_BG, gridcolor="rgba(255,255,255,0.25)", zerolinecolor="rgba(255,255,255,0.35)"),
    )
    fig3.update_traces(opacity=0.9)
    fig3.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=10))
    if use_log:
        fig3.update_scenes(xaxis_type="log", yaxis_type="log", zaxis_type="log")

    st.plotly_chart(fig3, use_container_width=True)

    # ======= NUEVO: Estadísticas descriptivas (después del 3D) =======
    st.markdown("### 📐 Estadísticas descriptivas (variables 3D seleccionadas)")
    cols_stats = [x3, y3, z3]
    if size_col != "(fijo)" and size_col in df3.columns:
        cols_stats.append(size_col)

    # Global
    t_stats1, t_stats2 = st.tabs(["Global", "Por clase"])
    with t_stats1:
        desc = df3[cols_stats].describe(percentiles=[0.25, 0.5, 0.75]).T
        desc = desc.rename(columns={
            "count":"n", "mean":"media", "std":"std",
            "min":"min", "25%":"q1", "50%":"mediana", "75%":"q3", "max":"max"
        })
        st.dataframe(desc.round(4), use_container_width=True)

        # Correlación
        if len(cols_stats) >= 2:
            corr = df3[cols_stats].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de correlación (Pearson)")
            style_plot(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)

    # Por clase
    with t_stats2:
        grp = df3.groupby("label")[cols_stats].agg(
            n=(""+cols_stats[0], "count")  # truco para crear al menos 1 col; reemplazamos después
        )
        # reconstruir con múltiples métricas
        agg_dict = {c: ["count", "mean", "std", "min", "median", "max"] for c in cols_stats}
        by_label = df3.groupby("label")[cols_stats].agg(agg_dict)
        # Aplanar columnas MultiIndex
        by_label.columns = [f"{c}_{stat}" for c, stat in by_label.columns]
        st.dataframe(by_label.round(4), use_container_width=True)

st.markdown("---")

# --- Entrenamiento
st.subheader("🧠 Entrenar modelo (TESS)")
if "trained" not in st.session_state:
    st.session_state["trained"] = False

train_clicked = st.button("Entrenar ahora", type="primary")
if train_clicked:
    try:
        with st.spinner("Entrenando y calibrando..."):
            # Con solo TESS, usamos split estratificado interno
            pipe, metrics, le = train_model(cat, test_subset=None)

            st.session_state["model"] = pipe
            st.session_state["metrics"] = metrics
            st.session_state["label_encoder"] = le
            st.session_state["trained"] = True

            buf = io.BytesIO()
            dump({"model": pipe, "features_num": FEATURES_NUM, "features_cat": FEATURES_CAT, "label_encoder": le}, buf)
            st.session_state["model_blob"] = buf.getvalue()

        st.success("Modelo entrenado ✅ (split interno estratificado en TESS).")
    except Exception as e:
        st.error("Ocurrió un error durante el entrenamiento.")
        st.exception(e)

if st.session_state["trained"]:
    m = st.session_state["metrics"]
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("F1 (CONFIRMED)", f"{m['report'].get('CONFIRMED', {}).get('f1-score', 0):.3f}")
    with colB:
        st.metric("F1 macro", f"{m['report'].get('macro avg', {}).get('f1-score', 0):.3f}")
    with colC:
        st.metric("AUC OvR (macro)", "-" if m["auc_macro_ovr"] is None else f"{m['auc_macro_ovr']:.3f}")

    cm = np.array(m["confusion_matrix"])
    fig_cm = px.imshow(cm, text_auto=True, x=m["labels"], y=m["labels"], aspect="auto",
                       labels=dict(x="Predicho", y="Real"))
    style_plot(fig_cm)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.download_button("⬇️ Descargar modelo (.joblib)", data=st.session_state["model_blob"],
                       file_name="exoplanet_model_tess.joblib")
st.markdown("---")

# --- Predicción demo + Visualizaciones
st.subheader("🔮 Predicción rápida (TESS)")
if st.session_state.get("trained", False):
    pipe = st.session_state["model"]
    le = st.session_state.get("label_encoder")
    class_names = list(le.classes_) if le is not None else [str(c) for c in getattr(pipe, "classes_", [])]

    # === Muestra 10 ejemplos
    show = cat.sample(min(10, len(cat)), random_state=42)
    X_show = show[FEATURES_NUM + ["mission"]]
    proba_show = pipe.predict_proba(X_show)
    preds_idx_show = pipe.predict(X_show)
    preds_show = le.inverse_transform(preds_idx_show) if le is not None else preds_idx_show

    out = show[["mission", "source_id"]].copy()
    out["pred"] = preds_show
    for i, cls in enumerate(class_names):
        out[f"P({cls})"] = proba_show[:, i]
    st.dataframe(out, use_container_width=True)

    # === Predicciones sobre TODO TESS (para visualizaciones y export)
    Xall = cat[FEATURES_NUM + ["mission"]]
    proba_all = pipe.predict_proba(Xall)
    pred_all_idx = pipe.predict(Xall)
    pred_all = le.inverse_transform(pred_all_idx) if le is not None else pred_all_idx

    pred_df = cat[["mission", "source_id"]].copy()
    pred_df["pred"] = pred_all
    for i, cls in enumerate(class_names):
        pred_df[f"P({cls})"] = proba_all[:, i]

    # ---------- Snapshot para el LLM (resumen de predicciones + métricas)
    pred_counts = pred_df["pred"].value_counts().to_dict()
    mean_probs = {col: float(pred_df[col].mean()) for col in pred_df.columns if col.startswith("P(")}
    top_conf = pred_df.sort_values("P(CONFIRMED)", ascending=False).head(5)[
        ["mission","source_id","P(CONFIRMED)"]
    ].to_dict(orient="records")
    top_fp   = pred_df.sort_values("P(FALSE POSITIVE)", ascending=False).head(5)[
        ["mission","source_id","P(FALSE POSITIVE)"]
    ].to_dict(orient="records")

    st.session_state["llm_context"] = {
        "pred_summary": {
            "counts": pred_counts,
            "mean_probs": mean_probs,
            "top_confirmed": top_conf,
            "top_false_positive": top_fp,
        },
        "metric_summary": st.session_state.get("metrics", {})
    }

    # ---------- Visualizaciones
    t1, t2, t3 = st.tabs(["📊 Distribución de clases", "📈 Histogramas de probabilidades", "🗺️ Mapa 2D por predicción"])

    # 1) Conteos (global TESS)
    with t1:
        cnt_global = pred_df["pred"].value_counts().rename_axis("clase").reset_index(name="count")
        fig_bar_pred = px.bar(
            cnt_global, x="clase", y="count", text_auto=True, title="Conteo de predicciones (TESS)",
            color="clase", color_discrete_map=COLOR_MAP
        )
        style_plot(fig_bar_pred)
        st.plotly_chart(fig_bar_pred, use_container_width=True)

    # 2) Histogramas por clase
    with t2:
        col_a, col_b = st.columns(2)
        mid = max(1, len(class_names) // 2)
        for container, names in [(col_a, class_names[:mid]), (col_b, class_names[mid:])]:
            with container:
                for cls in names:
                    fig_h = px.histogram(
                        pred_df, x=f"P({cls})", nbins=40, title=f"Distribución de P({cls})",
                        color_discrete_sequence=["#CFE9FF"]  # tono claro
                    )
                    style_plot(fig_h)
                    st.plotly_chart(fig_h, use_container_width=True)

    # 3) Scatter 2D coloreado por clase predicha
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            x_vis = st.selectbox("Eje X (2D)", FEATURES_NUM, index=0, key="pred_x2d")
        with c2:
            y_vis = st.selectbox("Eje Y (2D)", FEATURES_NUM, index=min(1, len(FEATURES_NUM)-1), key="pred_y2d")

        vis_df = cat[[x_vis, y_vis]].copy()
        vis_df["pred"] = pred_all
        vis_df["source_id"] = cat["source_id"].values

        max_pts = 10000
        if len(vis_df) > max_pts:
            vis_df = vis_df.sample(max_pts, random_state=42)

        fig_pred_sc = px.scatter(
            vis_df, x=x_vis, y=y_vis, color="pred",
            opacity=0.88, hover_data=["source_id"], title="Mapa 2D de predicciones (TESS)",
            color_discrete_map=COLOR_MAP
        )
        style_plot(fig_pred_sc)
        st.plotly_chart(fig_pred_sc, use_container_width=True)

    # Botón y descarga CSV
    if st.button("Generar archivo de predicciones (TESS)"):
        st.session_state["pred_export_csv"] = pred_df.to_csv(index=False).encode("utf-8")
        st.success("Predicciones generadas. ¡Listas para descargar!")

    if "pred_export_csv" in st.session_state:
        st.download_button("⬇️ Descargar predicciones (CSV)",
                           data=st.session_state["pred_export_csv"],
                           file_name="predicciones_tess.csv")
st.markdown("---")

# =======================
# 🤖 Copiloto Grok (xAI)
# =======================
st.subheader("🤖 Copiloto científico (Grok, xAI)")

def ensure_agent():
    if "grok" not in st.session_state:
        try:
            st.session_state["grok"] = GrokAgent()
            st.success("Grok inicializado.")
        except Exception as e:
            st.error(f"No se pudo inicializar Grok: {e}")
            st.stop()

ensure_agent()

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for role, text in st.session_state["chat"]:
    with st.chat_message(role):
        st.write(text)

prompt = st.chat_input("Haz una pregunta o pide una explicación (p. ej., '¿Por qué este objeto parece FP?')")
if prompt:
    st.session_state["chat"].append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    pipe = st.session_state.get("model")
    le = st.session_state.get("label_encoder")
    class_names = list(le.classes_) if le is not None else (
        [str(c) for c in getattr(pipe, "classes_", [])] if pipe is not None else ["CONFIRMED","CANDIDATE","FALSE POSITIVE"]
    )

    # ---- contexto para el LLM (TESS-only + snapshot)
    context_hint = {
        "dataset": "TESS only",
        "has_model": bool(st.session_state.get("trained", False)),
        "columns": list(cat.columns),
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "classes": class_names,
        "examples": 5,
        "viz_suggestion_guidelines": "Evita usar azul oscuro para puntos; sugiere X/Y/Z y rangos útiles.",
        **(st.session_state.get("llm_context") or {})
    }

    agent = st.session_state["grok"]
    data = agent.run(prompt, context_hint=context_hint)

    action = data.get("action", "NONE")
    args = data.get("args", {})
    narrative = data.get("narrative", {})

    tool_output = None
    try:
        if action == "METRICS":
            if not st.session_state.get("trained", False):
                tool_output = {"error": "No hay modelo entrenado."}
            else:
                m = st.session_state["metrics"]
                tool_output = {
                    "f1_confirmed": m["report"].get("CONFIRMED", {}).get("f1-score", 0),
                    "f1_macro": m["report"].get("macro avg", {}).get("f1-score", 0),
                    "auc_macro_ovr": m.get("auc_macro_ovr", None),
                }

        elif action == "QUERY_DF":
            df = cat.copy()
            flt = args.get("filters", {})
            for col, cond in flt.items():
                if col in df.columns and isinstance(cond, dict):
                    for op, val in cond.items():
                        if op == ">":   df = df[df[col] > float(val)]
                        elif op == ">=": df = df[df[col] >= float(val)]
                        elif op == "<":   df = df[df[col] < float(val)]
                        elif op == "<=": df = df[df[col] <= float(val)]
                        elif op == "==": df = df[df[col] == val]
            tool_output = df.head(int(args.get("limit", 20)))[
                ["mission", "source_id", "label", "radius_re", "orbital_period", "depth_ppm"]
            ].to_dict(orient="records")

        elif action == "EXPLAIN_CASE":
            sid = args.get("source_id", None)
            df = cat.copy()
            if sid: df = df[df["source_id"].astype(str) == str(sid)]
            if df.empty or not st.session_state.get("trained", False):
                tool_output = {"error": "No se encontró el caso o no hay modelo entrenado."}
            else:
                row = df.iloc[[0]]
                X = row[FEATURES_NUM + ["mission"]]
                proba = pipe.predict_proba(X)[0]
                pred_idx = pipe.predict(X)[0]
                pred_name = le.inverse_transform([pred_idx])[0] if le is not None else str(pred_idx)
                tool_output = {"pred": pred_name, "proba": dict(zip(class_names, [float(x) for x in proba]))}

        elif action == "PLOT":
            df = cat.copy()
            x = args.get("x", "radius_re"); y = args.get("y", "orbital_period")
            fig = px.scatter(df, x=x, y=y, color="label", opacity=0.88, hover_data=["source_id"],
                             color_discrete_map=COLOR_MAP, title="Gráfico solicitado (TESS)")
            style_plot(fig)
            st.plotly_chart(fig, use_container_width=True)
            tool_output = {"ok": True}
        else:
            tool_output = {}
    except Exception as e:
        tool_output = {"error": str(e)}

    with st.chat_message("assistant"):
        if narrative:
            st.markdown(f"**TL;DR:** {narrative.get('tldr','')}")
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f"**Clase:** {narrative.get('class','UNKNOWN')}")
            with cols[1]:
                st.markdown(f"**Confianza:** {narrative.get('confidence','-')}")
            with cols[2]:
                st.markdown("**Acciones:** " + ", ".join(narrative.get("next_steps", []) or []))
            if narrative.get("details"):
                st.markdown("**Detalles:**")
                st.markdown("\n".join([f"- {d}" for d in narrative["details"]]))
            if narrative.get("risks"):
                st.markdown("**Riesgos / cautelas:**")
                st.markdown("\n".join([f"- {r}" for r in narrative["risks"]]))
        if action and action != "NONE":
            st.markdown(f"**Acción ejecutada:** `{action}`")
            if tool_output:
                st.json(tool_output)

    st.session_state["chat"].append(("assistant", narrative.get("tldr", "(respuesta generada)")))
