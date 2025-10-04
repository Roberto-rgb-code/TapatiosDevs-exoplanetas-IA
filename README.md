Aquí está tu README completo y limpio (sin API keys expuestas):
markdown# A World Away: Hunting for Exoplanets with AI (TapatiosDevs)

Aplicación **Streamlit** para explorar candidatos a exoplanetas (TESS) y entrenar un modelo de **clasificación multiclase** (XGBoost calibrado) que predice:
- `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE` (y variantes si existen en datos).

Incluye:
- Exploración **2D y 3D** con paleta accesible (sin azules oscuros que se pierdan en el fondo).
- **Estadísticas descriptivas** (globales, por clase y correlaciones) justo después del gráfico 3D.
- Un **copiloto** (Grok, xAI) que interpreta predicciones, explica variables y sugiere visualizaciones.

---

## Requisitos

- **Python 3.10+** recomendado.
- Librerías (ver `requirements.txt`), típicamente:
  - `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `xgboost`, `python-dotenv`, `joblib`
  - SDK de xAI (`xai_sdk`) para el copiloto Grok.

> Si usas Streamlit Cloud, se instalarán automáticamente desde `requirements.txt`.

---

## Variables de entorno

Crea un archivo `.env` en la raíz (no se sube por seguridad):
```env
# Modelo LLM de xAI
GROK_MODEL=grok-4-fast-reasoning

# Clave de xAI (necesaria para el copiloto)
XAI_API_KEY=your-xai-api-key-here
En Streamlit Cloud coloca XAI_API_KEY como secrets en la UI del proyecto (Settings → Secrets), o en .streamlit/secrets.toml:
tomlXAI_API_KEY = "your-xai-api-key-here"
GROK_MODEL = "grok-4-fast-reasoning"

Datos
La app está configurada solo para TESS.

Puedes subir CSV adicionales desde el sidebar (⚙️ Datos → Subir CSV).
Si tienes datos locales, colócalos en data/. Por defecto, el .gitignore excluye subcarpetas pesadas (data/raw, data/interim, data/processed).
Si quieres versionar CSV pequeños de ejemplo, comenta la regla correspondiente en .gitignore.


Cómo ejecutar en local
1. Clonar e instalar
bashgit clone https://github.com/Roberto-rgb-code/TapatiosDevs-exoplanetas-IA.git
cd TapatiosDevs-exoplanetas-IA

# (opcional) crear venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
2. Variables de entorno
Crea .env con tus claves (ver sección anterior).
3. Levantar la app
bashstreamlit run app.py
4. Navegar
Abre el link que Streamlit imprime en la terminal (por defecto http://localhost:8501).

Despliegue en Streamlit Cloud

Sube este repo a GitHub (incluyendo requirements.txt y NO subas .env).
Ve a share.streamlit.io y conecta tu repo.
En Advanced settings → Secrets, añade:

XAI_API_KEY
GROK_MODEL (opcional)


Deploy. Podrás subir CSV desde la UI de la app.


Uso rápido

Entrenar: Botón "Entrenar ahora".
Explorar:

Gráficos 2D y 3D; activa escala log si lo necesitas.
Revisa estadísticas descriptivas (resumen, cuantiles y correlación) debajo del gráfico 3D.


Predicción rápida: muestra probabilidades por clase y permite descargar un CSV con predicciones.
Copiloto: pregunta cosas como:

"Explica por qué este objeto parece FP"
"Sugiere un gráfico para separar candidates vs confirmed con teff y depth_ppm"




Preguntas frecuentes
1) No veo datos al cargar.
Verifica que tus CSV tengan la columna mission y que contengan filas con TESS. O sube archivos desde el sidebar.
2) Error con xAI/Grok.
Asegúrate de definir XAI_API_KEY en .env (local) o en Secrets (Cloud). Si no usas el copiloto, la app base funciona igual.
3) ¿Qué modelo de ML usan?
Un XGBoost multiclase calibrado con isotonic dentro de un Pipeline de scikit-learn (estandarización para numéricos + OneHot para categóricas). Mira models/pipeline.py.

Licencia
MIT (o la que prefieras)