# chatbot/grok_agent.py
from __future__ import annotations
import os, json, re
from typing import Any, Dict, List
from xai_sdk import Client
from xai_sdk.chat import system, user

GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")

# --- Glosario centralizado (nombre -> definición breve + cómo interpretarlo)
VARIABLES_GLOSSARY = {
    "depth_ppm": "Profundidad del tránsito (ppm): caída de brillo durante el tránsito; mayor profundidad suele indicar objetos más grandes o eventos de eclipse.",
    "snr": "Relación señal-ruido (SNR): calidad de la detección; valores altos suelen ser más confiables.",
    "radius_re": "Radio del objeto (R⊕): tamaño en radios de la Tierra; ayuda a distinguir sub-Neptunos / Júpiteres calientes.",
    "duration_hours": "Duración del tránsito (h): tiempo en horas; depende de geometría, período y tamaño estelar.",
    "orbital_period": "Período orbital (días): tiempo entre tránsitos; relaciona distancia orbital y temperatura del planeta.",
    "insol": "Irradiación relativa: flujo recibido respecto a la Tierra; alto → objeto/órbita más caliente.",
    "teff": "Temperatura efectiva estelar (K): determina el color/luminosidad de la estrella anfitriona.",
    "star_rad_rs": "Radio estelar (R☉): tamaño de la estrella; influye en la profundidad aparente del tránsito.",
    "mag": "Magnitud aparente: brillo observado; más bajo = más brillante (mejor para seguimiento).",
    "mission": "Misión de origen del candidato; en esta app trabajamos con TESS."
}

# --- Plantilla de objetivo del modelo (para reforzar 'de qué trata la predicción')
MODEL_OBJECTIVE_SENTENCE = (
    "El modelo clasifica objetos TESS en {'CONFIRMED','CANDIDATE','FALSE POSITIVE'} "
    "a partir de rasgos de tránsito y del sistema (p. ej., depth_ppm, snr, radius_re), "
    "devolviendo probabilidades calibradas por clase."
)

SYSTEM_PROMPT = f"""Eres un copiloto astrofísico (ES). Tu meta es explicar con claridad:
- Qué está prediciendo el modelo (objetivo, clases y probabilidades).
- Qué significan las variables y cómo influyen.
- Qué visualizaciones usar para entender o validar casos.
- Qué limitaciones y próximos pasos recomendar.

SIEMPRE responde con **UN JSON VÁLIDO** con esta forma mínima:

{{
  "action": "<NONE|EXPLAIN_CASE|QUERY_DF|PLOT|METRICS>",
  "args": {{ }},
  "narrative": {{
    "tldr": "resumen breve y claro",
    "class": "CONFIRMED|CANDIDATE|FALSE POSITIVE|UNKNOWN",
    "confidence": "alto|medio|bajo",
    "details": ["bullets ..."],
    "risks": ["riesgos/limitaciones ..."],
    "next_steps": ["acciones prácticas ..."]
  }},
  "viz_suggestions": {{
    "plots": [
      {{"kind":"scatter2d","x":"radius_re","y":"depth_ppm","mission":null}},
      {{"kind":"scatter3d","x":"radius_re","y":"orbital_period","z":"duration_hours"}}
    ],
    "filters": [{{"column":"mission","op":"in","value":["TESS"]}}],
    "notes": "Evita azules oscuros para puntos; prioriza alto contraste"
  }},
  "glossary": [
    {{"name":"depth_ppm","explain":"Profundidad del tránsito (ppm) ..."}},
    {{"name":"snr","explain":"Relación señal-ruido ..."}}
  ]
}}

Reglas:
- Usa SOLO columnas reales del contexto.
- Si el usuario pide explicar un objeto concreto → action="EXPLAIN_CASE" (args con source_id si se infiere).
- Si pide métricas globales → action="METRICS".
- Si pide ver filas → action="QUERY_DF" (usa >, >=, <, <=, ==).
- Si pide un gráfico → action="PLOT" con {{x,y,(z opcional)}}.
- Si solo pide “interpretar resultados” en general: NO llames herramientas; usa "pred_summary" y "metric_summary" del contexto para la narrativa.
- Incluye SIEMPRE una oración que diga explícitamente el objetivo del modelo (por ejemplo: "{MODEL_OBJECTIVE_SENTENCE}").
- Sé conciso: 1–2 frases en tldr; 3–6 bullets en details.
- Evita paletas de puntos azul oscuro (se confunden con el fondo).
"""

ALLOWED_ACTIONS = {"NONE","EXPLAIN_CASE","QUERY_DF","PLOT","METRICS"}

def _coerce_json(s: str) -> Dict[str, Any]:
    """Intenta extraer un JSON válido desde el texto del modelo."""
    try:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass
    try:
        blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S|re.I)
        for b in blocks:
            try: return json.loads(b)
            except Exception: continue
    except Exception:
        pass
    # Fallback seguro
    tldr = (s or "").strip().replace("\n"," ")
    if len(tldr) > 400: tldr = tldr[:400] + "…"
    return {
        "action":"NONE","args":{},
        "narrative":{
            "tldr": tldr or "(sin contenido)",
            "class":"UNKNOWN","confidence":"bajo",
            "details":[], "risks":[], "next_steps":[]
        },
        "viz_suggestions":{"plots":[],"filters":[],"notes":""},
        "glossary":[]
    }

def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza estructura mínima y limita longitudes."""
    if not isinstance(d, dict): d = {}
    action = str(d.get("action","NONE")).upper()
    if action not in ALLOWED_ACTIONS: action = "NONE"
    args = d.get("args") or {}
    if not isinstance(args, dict): args = {}

    nar = d.get("narrative") or {}
    if not isinstance(nar, dict): nar = {}
    nar.setdefault("tldr","")
    nar.setdefault("class","UNKNOWN")
    nar.setdefault("confidence","bajo")
    for k in ("details","risks","next_steps"):
        v = nar.get(k, [])
        nar[k] = [str(x) for x in (v if isinstance(v, list) else [])][:6]

    viz = d.get("viz_suggestions") or {}
    if not isinstance(viz, dict): viz = {}
    viz.setdefault("plots", []); viz.setdefault("filters", []); viz.setdefault("notes", "")

    glossary = d.get("glossary") or []
    if not isinstance(glossary, list): glossary = []
    glossary = glossary[:10]

    return {"action":action, "args":args, "narrative":nar, "viz_suggestions":viz, "glossary":glossary}

def _auto_glossary(columns: List[str]) -> List[Dict[str,str]]:
    out = []
    seen = set()
    for col in columns:
        if col in VARIABLES_GLOSSARY and col not in seen:
            out.append({"name": col, "explain": VARIABLES_GLOSSARY[col]})
            seen.add(col)
    # asegura claves típicas si existen
    for k in ("depth_ppm","snr","radius_re","duration_hours","orbital_period"):
        if k in VARIABLES_GLOSSARY and (k not in seen):
            out.append({"name": k, "explain": VARIABLES_GLOSSARY[k]})
            seen.add(k)
    return out[:10]

def _enrich(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Completa objetivo del modelo, sugerencias de plot y glosario si faltan."""
    if not ctx: return payload
    feats = list(ctx.get("features_num", []))
    cols  = list(ctx.get("columns", []))
    pred_summary = ctx.get("pred_summary") or {}
    metric_summary = ctx.get("metric_summary") or {}

    # 1) Asegurar que el TL;DR mencione el objetivo del modelo
    nar = payload.get("narrative", {})
    tldr = nar.get("tldr","").strip()
    if "clasifica objetos TESS" not in tldr:
        prefix = "Objetivo: " + MODEL_OBJECTIVE_SENTENCE
        nar["tldr"] = (prefix + " " + tldr).strip() if tldr else prefix
    payload["narrative"] = nar

    # 2) Viz por defecto si no hay acción y no hay plots
    if payload.get("action") == "NONE":
        viz = payload.get("viz_suggestions", {})
        plots = viz.get("plots") or []
        if not plots:
            if len(feats) >= 2:
                plots.append({"kind":"scatter2d","x":feats[0],"y":feats[1],"mission":["TESS"]})
            if len(feats) >= 3:
                plots.append({"kind":"scatter3d","x":feats[0],"y":feats[1],"z":feats[2]})
            viz["plots"] = plots
        notes = viz.get("notes","")
        if "azul" not in notes.lower():
            viz["notes"] = (notes + " Evita azules oscuros para puntos; usa alto contraste.").strip()
        payload["viz_suggestions"] = viz

    # 3) Glosario auto-llenado si viene vacío
    glossary = payload.get("glossary") or []
    if not glossary:
        candidate_cols = [c for c in feats if c in VARIABLES_GLOSSARY] + \
                         [c for c in cols  if c in VARIABLES_GLOSSARY]
        payload["glossary"] = _auto_glossary(candidate_cols or list(VARIABLES_GLOSSARY.keys()))

    # 4) Añadir bullets interpretativos usando resúmenes (si existen)
    details = payload["narrative"].get("details", [])
    if pred_summary and len(details) < 6:
        counts = pred_summary.get("counts", {})
        if counts:
            top_class = max(counts, key=counts.get)
            details.append(f"Distribución de predicciones (TESS): clase más frecuente = {top_class} (conteos: {counts}).")
    if metric_summary and len(details) < 6:
        f1m = metric_summary.get("report", {}).get("macro avg", {}).get("f1-score", None)
        if f1m is not None:
            details.append(f"Macro-F1 del modelo ≈ {f1m:.3f}, útil para comparar balance entre clases.")
    payload["narrative"]["details"] = details[:6]

    return payload

class GrokAgent:
    def __init__(self, api_key: str | None = None, model: str = GROK_MODEL, timeout: int = 120, temperature: float = 0.2):
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta XAI_API_KEY en entorno.")
        self.client = Client(api_key=api_key, timeout=timeout)
        self.chat = self.client.chat.create(model=model, temperature=temperature)
        self.chat.append(system(SYSTEM_PROMPT))

    def run(self, user_text: str, context_hint: dict | None = None) -> dict:
        # Empaqueta un contexto compacto con resúmenes si existen
        ctx = context_hint or {}
        compact_ctx = {
            "has_model": bool(ctx.get("has_model")),
            "columns": list(ctx.get("columns", []))[:100],
            "features_num": list(ctx.get("features_num", []))[:50],
            "features_cat": list(ctx.get("features_cat", []))[:50],
            "classes": list(ctx.get("classes", []))[:20],
            "pred_summary": ctx.get("pred_summary", {}),
            "metric_summary": ctx.get("metric_summary", {}),
            "style_notes": "Fondo azul medio; evita puntos azul oscuro; usa alto contraste."
        }
        prompt = f"{user_text.strip()}\n\n[contexto]\n{json.dumps(compact_ctx, ensure_ascii=False)}"
        self.chat.append(user(prompt))
        resp = self.chat.sample()
        raw = _coerce_json(getattr(resp, "content", str(resp)) or "")
        clean = _sanitize(raw)
        enriched = _enrich(clean, compact_ctx)
        return enriched
