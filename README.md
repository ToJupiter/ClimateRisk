✅ **Climate Risk Impact on Firm Value: A Causal Analysis using Double ML** 🌍📈

---

### 🎯 Goal  
**Does talking more about climate risk in reports *cause* changes in firm value (Tobin’s Q)?**  
→ Answered via **causal ML** with confounders & ESG moderation.

---

### 📁 Structure (Key Files)

| Icon | File | Purpose |
|------|------|---------|
| 🖥️ | `main.py` | CLI to run pipeline |
| 📄 | `pdf_processor.py` | PDF → Text (PyMuPDF + multiprocessing) |
| 🔍 | `tfidf_calculator.py` | Compute `ClimateRiskScore` from keywords |
| 🧠 | `causal_analysis.py` | EconML DML models + SHAP + plots |
| ⚙️ | `config.py` | Paths, keywords, params |
| 📦 | `requirements.txt` | All deps (incl. `econml`, `xgboost`, `shap`) |

---

### 🔄 Pipeline

1. **PDF → TXT**  
   Extract text from annual reports → `output_txt/`

2. **TF-IDF → ClimateRiskScore**  
   Sum TF-IDF of climate keywords → single score per firm

3. **Causal DML (4 Cases)**  
   - Case 1: Baseline (no confounders/moderator)  
   - Case 2: + Confounders (size, leverage, GDP, etc.)  
   - Case 3: + Moderator (ESG score)  
   - ✅ **Case 4: Full model (confounders + ESG)** ← Main analysis

---

### 🛠️ Install (uv + Python 3.9)

```bash
git clone https://github.com/ToJupiter/ClimateRisk.git
cd ClimateRisk
uv venv -p 3.9 && source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

---

### ▶️ Run

```bash
python main.py
```
→ Choose:  
1️⃣ PDF → TXT  
2️⃣ TF-IDF Scores  
3️⃣ 🚀 **Run Causal ML + Plots**

---

### 🧰 Key Tools

- `PyMuPDF` → Fast PDF extraction  
- `scikit-learn` → TF-IDF, ML models  
- `EconML` → Double ML for causal effects  
- `XGBoost` → Base learner in DML  
- `SHAP` → Feature importance (fixed ✅)  
- `Matplotlib/Seaborn` → Visuals

---

### 📈 Output

- `distribution_plots.png`  
- `cate_by_esg.png` (Moderator effect)  
- `shap_feature_importance.png` ✅ *(now fixed with TreeExplainer)*

---

### 📜 License: MIT

---

✅ **Concise. Visual. Actionable.**  
Let me know if you want the README as Markdown file or need CI/CD setup!
