# A World Away: Hunting for Exoplanets with AI (TapatiosDevs)

**Live app (deployed):**  
**https://tapatiosdevs-exoplanetas-ia-stybmqnqfukrfg9wwvtwuu.streamlit.app/**

**Primary data source (TESS / TOI table):**  
**https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI**

Streamlit application to explore TESS exoplanet **candidates** and train a **multiclass classifier** (calibrated XGBoost) that predicts:
- `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE` (and textual variants present in the data).

Includes:
- **2D and 3D exploration** with an accessible palette (no dark blues that blend into the background).
- **Descriptive statistics** (global, by class, and correlations) right after the 3D plot.
- An **AI Copilot** (Grok, xAI) that interprets predictions, explains variables, and suggests visualizations.

---

## Highlights

- 🚀 **Deployed & public:** The project is already live at the URL above.
- 📊 **TESS-only workflow:** The app filters to **TESS** rows and operates on the TOI dataset.
- 🧠 **Calibrated ML:** Multiclass XGBoost with probability calibration for realistic confidence.
- 🧭 **Explainable UI:** Concept notes on why variables matter + charts to inspect separability.

---

## Requirements

- **Python 3.10+** recommended.
- Core libraries (see `requirements.txt`), typically:
  - `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `xgboost`, `python-dotenv`, `joblib`
  - xAI SDK for Grok (e.g., `xai_sdk`) if you want the Copilot enabled.

> On Streamlit Cloud, dependencies are installed automatically from `requirements.txt`.

---

## Environment Variables

Create a `.env` file at the project root (do **not** commit it):

```env
# xAI LLM model
GROK_MODEL=grok-4-fast-reasoning

# xAI API key (required for the Copilot)
XAI_API_KEY=your-xai-api-key-here
On Streamlit Cloud, set secrets via the UI (Settings → Secrets) or in .streamlit/secrets.toml:

toml
Copiar código
XAI_API_KEY = "your-xai-api-key-here"
GROK_MODEL = "grok-4-fast-reasoning"
Data
The app is configured to operate only on TESS.

You can upload extra CSVs from the sidebar (⚙️ Data → Upload additional CSV files).

If you have local sample data, place CSVs in data/. The .gitignore typically excludes heavy subfolders like data/raw, data/interim, data/processed.
If you want to version small example CSVs, comment out the relevant ignore rules.

Reference dataset (live source used):
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI

Run Locally
Clone & install

bash
Copiar código
git clone https://github.com/Roberto-rgb-code/TapatiosDevs-exoplanetas-IA.git
cd TapatiosDevs-exoplanetas-IA

# (optional) create venv
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
Environment variables
Create .env with your keys (see section above).

Start the app

bash
Copiar código
streamlit run app.py
Open in browser
Use the link printed by Streamlit (default: http://localhost:8501).

Deploy on Streamlit Cloud
Push this repo to GitHub (include requirements.txt; do not commit .env).

Go to share.streamlit.io and connect your repo.

In Advanced settings → Secrets, add:

XAI_API_KEY

GROK_MODEL (optional, falls back to grok-4-fast-reasoning)

Deploy. You can upload CSVs from the app UI once it’s running.

Public deployment (already live):
https://tapatiosdevs-exoplanetas-ia-stybmqnqfukrfg9wwvtwuu.streamlit.app/

Quick Start (in the App)
Train: Click “Train now” (TESS-only, internal stratified split).

Explore:

Use 2D and 3D selectors; enable log scale if helpful.

Check descriptive stats (summary, quantiles, correlations) below the 3D plot.

Quick Prediction: Shows class probabilities and lets you download a CSV of predictions.

Copilot (Grok): Ask questions like:

“Explain why this object looks like an FP.”

“Suggest a chart to separate candidates vs confirmed with teff and depth_ppm.”

FAQ
1) I see no data after loading.
Ensure your CSVs include a mission column and contain TESS rows. Or upload files via the sidebar.

2) xAI/Grok errors.
Define XAI_API_KEY in .env (local) or in Secrets (Cloud).
If you don’t use the Copilot, the core app still runs fine.

3) Which ML model is used?
A multiclass XGBoost with isotonic calibration, wrapped in a scikit-learn Pipeline
(standardization for numeric features + one-hot encoding for categorical features).
See models/pipeline.py.

Concept Notes (Why these variables?)
radius_re (planet radius): extreme radii (>30 R⊕) often indicate binaries/noise → FP.

depth_ppm (transit depth): must be coherent with planet/star sizes.

duration_hours (transit duration): very short → likely instrumental noise.

orbital_period (days): confirmed typically in ~1–50 days; extreme/unstable → FP.

insol, teff, star_rad_rs: physical consistency checks (irradiance, stellar type/size).

mission: contextualizes noise patterns (e.g., TESS vs. Kepler).

License
MIT (or your preferred license).
Feel free to adapt and extend for research or educational purposes.