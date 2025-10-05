# chatbot/grok_agent.py
from __future__ import annotations
import os, json, re
from typing import Any, Dict, List
from xai_sdk import Client
from xai_sdk.chat import system, user

GROK_MODEL = os.getenv("GROK_MODEL", "grok-4")

# =========================
# 📚 Glosario y sinónimos
# =========================
# Mapea columnas del NASA Exoplanet Archive (TESS) a las features usadas en el pipeline y sinónimos comunes.
VAR_GLOSSARY = {
    # --- Planeta / tránsito ---
    "orbital_period": {
        "aliases": ["period", "periodo", "período", "pl_orbper", "days", "días"],
        "from_table": "pl_orbper",
        "unit": "días",
        "explain": "Tiempo que tarda el planeta en completar una órbita. En tránsitos, periodos cortos generan más eventos observables."
    },
    "duration_hours": {
        "aliases": ["duration", "duración", "pl_trandurh", "hours", "horas"],
        "from_table": "pl_trandurh",
        "unit": "horas",
        "explain": "Duración del tránsito (inicio a fin). Tránsitos muy cortos o muy largos pueden indicar geometrías o contaminaciones atípicas."
    },
    "depth_ppm": {
        "aliases": ["depth", "profundidad", "trandept", "pl_trandep", "ppm"],
        "from_table": "pl_trandep",
        "unit": "ppm",
        "explain": "Cuánto se atenúa la luz de la estrella durante el tránsito. Profundidades más altas suelen indicar planetas más grandes o blends."
    },
    "radius_re": {
        "aliases": ["radius", "radio", "pl_rade", "re", "r_earth"],
        "from_table": "pl_rade",
        "unit": "R⊕",
        "explain": "Radio del planeta en radios-Tierra. Depende de la profundidad y del radio estelar asumido."
    },
    "insol": {
        "aliases": ["insolation", "irradiance", "pl_insol", "flux"],
        "from_table": "pl_insol",
        "unit": "F⊕",
        "explain": "Flujo incidente relativo a la Tierra; aproxima qué tanta energía recibe el planeta."
    },
    # --- Estrella ---
    "teff": {
        "aliases": ["teff", "temperatura", "st_teff", "effective temperature"],
        "from_table": "st_teff",
        "unit": "K",
        "explain": "Temperatura efectiva de la estrella. Afecta la profundidad esperada y el cálculo de insolación."
    },
    "star_rad_rs": {
        "aliases": ["stellar radius", "st_rad", "radio estelar", "rs", "r_sun"],
        "from_table": "st_rad",
        "unit": "R☉",
        "explain": "Radio de la estrella. Clave para convertir profundidad de tránsito a radio planetario."
    },
    "mag": {
        "aliases": ["tess mag", "st_tmag", "magnitud", "tmagtess"],
        "from_table": "st_tmag",
        "unit": "mag (TESS)",
        "explain": "Magnitud en banda TESS; afecta SNR y detectabilidad."
    },
    # --- Señal ---
    "snr": {
        "aliases": ["s/n", "signal to noise", "relación señal ruido", "snr"],
        "from_table": None,  # puede no venir directo en TOI; si existe en tu DF, se usará tal cual
        "unit": "adimensional",
        "explain": "Relación señal-ruido del tránsito. Más alto, más confiable (ojo a outliers por systematics)."
    },
}

# Mapa inverso de sinónimos → key estándar
ALIAS2KEY: Dict[str, str] = {}
for k, meta in VAR_GLOSSARY.items():
    ALIAS2KEY[k.lower()] = k
    for a in meta["aliases"]:
        ALIAS2KEY[a.lower()] = k

# =========================
# 🧭 Sistema conversacional
# =========================
SYSTEM_PROMPT = """Eres un copiloto astrofísico en español, amable y claro.
Puedes saludar, guiar al usuario por la plataforma, explicar variables (con sinónimos) y resumir resultados.
NO uses jerga innecesaria. Mantén un tono cercano y pedagógico.

SIEMPRE devuelve **UN JSON VÁLIDO** con esta forma:

{
  "action": "<NONE|EXPLAIN_CASE|QUERY_DF|PLOT|METRICS>",
  "args": { },
  "narrative": {
    "tldr": "resumen breve (1-2 frases)",
    "class": "CONFIRMED|CANDIDATE|FALSE POSITIVE|UNKNOWN",
    "confidence": "alto|medio|bajo",
    "details": ["bullets (3–6 máx)"],
    "risks": ["limitaciones o sesgos relevantes"],
    "next_steps": ["siguientes acciones o vistas a usar"]
  },
  "viz_suggestions": {
    "plots": [{"kind":"scatter2d","x":"radius_re","y":"depth_ppm"}],
    "filters": [{"column":"mission","op":"in","value":["TESS"]}],
    "notes": "Evita azules oscuros para puntos; usa alto contraste."
  }
}

REGLAS:
- Usa SOLO columnas reales del contexto (features/columns).
- Si el usuario saluda o pide ayuda/explique la plataforma → action="NONE", responde con narrativa y tips.
- Si pregunta “¿qué significa X?” o “explica la variable Y?” → incluye explicación breve (usa 'details').
- Si pide interpretar resultados globales y hay pred_summary/metric_summary → usa eso, sin llamar tools.
- Si pide un caso concreto → action="EXPLAIN_CASE" (args con source_id si se deduce).
- Si pide métricas → action="METRICS".
- Si pide filas con filtros → "QUERY_DF".
- Si pide un gráfico → "PLOT" con {x,y,(z opcional)}.
- Mantén “tldr” conciso; 3–6 bullets en “details”; evita párrafos largos.
- No inventes columnas: valida con 'columns' y 'features_*' del contexto.
- Evita usar azul oscuro para puntos (se mezcla con el tema de fondo).
"""

ALLOWED_ACTIONS = {"NONE","EXPLAIN_CASE","QUERY_DF","PLOT","METRICS"}

def _coerce_json(s: str) -> Dict[str, Any]:
    """Extrae JSON aunque venga envuelto en texto/código; fallback a narrativa simple."""
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
    t = (s or "").strip().replace("\n"," ")
    if len(t) > 400: t = t[:400] + "…"
    return {
        "action":"NONE","args":{},
        "narrative":{
            "tldr": t or "Hola, ¿en qué te ayudo? Puedo explicar las variables, guiarte por la app o interpretar resultados.",
            "class":"UNKNOWN","confidence":"bajo","details":[],"risks":[],"next_steps":[]
        },
        "viz_suggestions":{"plots":[],"filters":[],"notes":""}
    }

def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
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
    return {"action":action,"args":args,"narrative":nar,"viz_suggestions":viz}

def _best_var_key(query: str, available: List[str]) -> str | None:
    """Encuentra la variable pedida por el usuario usando sinónimos y valida que exista en el DF."""
    q = (query or "").lower()
    # Busca por alias
    for token in re.findall(r"[a-zA-Z_]+", q):
        key = ALIAS2KEY.get(token.lower())
        if key and (key in available):
            return key
    # match directo por contains con columnas disponibles
    for col in available:
        if col.lower() in q:
            return col
    return None

def _fill_defaults(payload: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not ctx: return payload
    cols = ctx.get("columns") or []
    feats = (ctx.get("features_num") or []) + (ctx.get("features_cat") or [])
    valid = set(cols) | set(feats)
    a = payload.get("action")
    args = payload.get("args", {})

    # Autocompletar PLOT x/y
    if a == "PLOT":
        x = args.get("x"); y = args.get("y")
        if not x or x not in valid:
            args["x"] = next((f for f in ctx.get("features_num", []) if f in valid), next(iter(valid), "radius_re"))
        if not y or y not in valid or y == args["x"]:
            y_cand = [f for f in ctx.get("features_num", []) if f != args["x"] and f in valid]
            args["y"] = y_cand[0] if y_cand else "orbital_period"
        payload["args"] = args

    # Limitar QUERY_DF
    if a == "QUERY_DF":
        try: lim = int(args.get("limit", 20))
        except Exception: lim = 20
        args["limit"] = max(1, min(lim, 200))
        payload["args"] = args

    # Sugerencias por defecto si no hay acción
    if a == "NONE":
        viz = payload.get("viz_suggestions", {})
        if isinstance(viz, dict) and not viz.get("plots"):
            basic = []
            feats_num = ctx.get("features_num", [])
            if len(feats_num) >= 2:
                basic.append({"kind":"scatter2d","x":feats_num[0],"y":feats_num[1]})
            if len(feats_num) >= 3:
                basic.append({"kind":"scatter3d","x":feats_num[0],"y":feats_num[1],"z":feats_num[2]})
            viz["plots"] = basic
            viz["notes"] = (viz.get("notes","") + " Evita azules oscuros; usa alto contraste.").strip()
            payload["viz_suggestions"] = viz
    return payload

def _compose_help_narrative(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Respuesta amable de onboarding/ayuda general."""
    has_model = ctx.get("has_model", False)
    details = [
        "Exploro candidatos TESS y predigo: **CONFIRMED**, **CANDIDATE** o **FALSE POSITIVE**.",
        "Gráficas recomendadas: 2D `radius_re` vs `depth_ppm` y 3D con `orbital_period`/`duration_hours`.",
        "Pídeme: *'¿qué significa depth?'*, *'explica teff'*, *'métricas del modelo'*, o *'grafica radius vs period'*.",
    ]
    if has_model:
        details.append("Tu modelo ya está entrenado: puedo interpretar probabilidades y casos concretos (EXPLAIN_CASE).")
    else:
        details.append("Aún no hay modelo entrenado: ve a **🧠 Entrenar modelo (TESS)** y pulsa *Entrenar ahora*.")
    return {
        "tldr": "¡Hola! Soy tu copiloto astrofísico. Te explico variables, gráficos y resultados, y te guío por la app.",
        "class": "UNKNOWN",
        "confidence": "alto",
        "details": details[:6],
        "risks": ["Las predicciones dependen de la calidad de datos y balance de clases."],
        "next_steps": ["Abre 'Predicción rápida (TESS)' para ver distribuciones", "Pídeme un gráfico o una explicación de variable"]
    }

def _compose_var_narrative(var_key: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    meta = VAR_GLOSSARY.get(var_key, {})
    label = var_key
    unit = meta.get("unit", "")
    from_table = meta.get("from_table", "")
    expl = meta.get("explain", "Variable del catálogo.")
    bullets = [
        f"**Nombre:** `{label}`  • **TOI column:** `{from_table or label}`  • **Unidad:** {unit or '—'}",
        f"**Qué es:** {expl}",
        "Impacto en la predicción: el modelo aprende patrones multivariados; por sí sola no decide la clase.",
    ]
    bullets.append("Tip de gráfica: usa un *scatter* con color por clase; evita azules oscuros para puntos.")
    return {
        "tldr": f"`{label}`: {expl}",
        "class": "UNKNOWN",
        "confidence": "alto",
        "details": bullets[:6],
        "risks": ["Cuidado con valores extremos y unidades/escala; revisa outliers."],
        "next_steps": [f"Grafica `{label}` vs `depth_ppm` o `orbital_period`", "Consulta histogramas de probabilidades"]
    }

def _compose_pipeline_narrative(ctx: Dict[str, Any]) -> Dict[str, Any]:
    bullets = [
        "Preprocesamiento: **StandardScaler** en numéricos y **OneHotEncoder** para `mission`.",
        "Modelo: **XGBoost** multiclase calibrado con **isotonic** (`CalibratedClassifierCV cv=3`).",
        "Entrenamiento: *stratified split* (o test externo si hubiera varias misiones).",
        "Métricas: reporte de clasificación, matriz de confusión y **AUC OvR (macro)**.",
        "Predicción: `predict_proba` → probabilidades por clase; salida principal = clase con mayor prob.",
    ]
    return {
        "tldr": "El pipeline estandariza/one-hot, entrena XGBoost multiclase y calibra probabilidades.",
        "class": "UNKNOWN",
        "confidence": "alto",
        "details": bullets[:6],
        "risks": ["Desbalance de clases puede afectar F1; calibra y revisa matriz de confusión."],
        "next_steps": ["Mira '🧠 Entrenar modelo (TESS)' y luego '🔮 Predicción rápida (TESS)'"]
    }

class GrokAgent:
    def __init__(self, api_key: str | None = None, model: str = GROK_MODEL, timeout: int = 120, temperature: float = 0.2):
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta XAI_API_KEY en entorno.")
        self.client = Client(api_key=api_key, timeout=timeout)
        self.chat = self.client.chat.create(model=model, temperature=temperature)
        self.chat.append(system(SYSTEM_PROMPT))

    def run(self, user_text: str, context_hint: dict | None = None) -> dict:
        ctx = context_hint or {}
        # ---- Detección simple de intención para respuestas naturales (sin llamar tools)
        text = (user_text or "").strip().lower()
        columns = [c.lower() for c in (ctx.get("columns") or [])]
        features = [c.lower() for c in (ctx.get("features_num") or []) + (ctx.get("features_cat") or [])]
        available = list(set(columns) | set(features))

        # ¿Saludo/ayuda?
        if re.search(r"\b(hola|buenas|qué onda|ayuda|help|cómo uso|como uso|guía|guiame|guíame)\b", text):
            payload = {
                "action": "NONE",
                "args": {},
                "narrative": _compose_help_narrative(ctx),
                "viz_suggestions": {
                    "plots": [{"kind":"scatter2d","x":"radius_re","y":"depth_ppm"}],
                    "filters": [{"column":"mission","op":"in","value":["TESS"]}],
                    "notes": "Evita azules oscuros para puntos; usa alto contraste."
                }
            }
            return _sanitize(payload)

        # ¿Pregunta de variable? (qué es, que significa, explica X)
        if re.search(r"(qué\s+es|que\s+es|qué\s+significa|que\s+significa|explica|definición de)\s+", text):
            key = _best_var_key(text, available)
            if key:
                payload = {
                    "action": "NONE",
                    "args": {},
                    "narrative": _compose_var_narrative(key, ctx),
                    "viz_suggestions": {
                        "plots": [{"kind":"scatter2d","x":key,"y":"depth_ppm"}],
                        "filters": [{"column":"mission","op":"in","value":["TESS"]}],
                        "notes": "Evita azules oscuros; prioriza contraste."
                    }
                }
                return _sanitize(payload)

        # ¿Pipeline / cómo funciona / objetivo?
        if re.search(r"(pipeline|cómo funciona|como funciona|objetivo|qué predice|que predice|modelo)\b", text):
            payload = {
                "action": "NONE",
                "args": {},
                "narrative": _compose_pipeline_narrative(ctx),
                "viz_suggestions": {
                    "plots": [{"kind":"scatter3d","x":"radius_re","y":"orbital_period","z":"duration_hours"}],
                    "filters": [{"column":"mission","op":"in","value":["TESS"]}],
                    "notes": "Usa alto contraste en puntos; añade hover con `source_id`."
                }
            }
            return _sanitize(payload)

        # Si no hay una intención “charla/ayuda” detectada, mandamos al modelo LLM con contexto compacto
        compact_ctx = {
            "has_model": bool(ctx.get("has_model")),
            "columns": list(ctx.get("columns", []))[:100],
            "features_num": list(ctx.get("features_num", []))[:50],
            "features_cat": list(ctx.get("features_cat", []))[:50],
            "classes": list(ctx.get("classes", []))[:20],
            "pred_summary": ctx.get("pred_summary", {}),
            "metric_summary": ctx.get("metric_summary", {}),
            "style_notes": "Fondo azul medio; evita puntos azul oscuro."
        }

        prompt = f"{user_text.strip()}\n\n[contexto]\n{json.dumps(compact_ctx, ensure_ascii=False)}"
        self.chat.append(user(prompt))
        resp = self.chat.sample()
        raw = _coerce_json(getattr(resp, "content", str(resp)) or "")
        clean = _sanitize(raw)
        final = _fill_defaults(clean, compact_ctx)
        return final
