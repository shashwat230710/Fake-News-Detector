
# Fake News Detection + Summarizer — Full Roadmap & Starter Kit

This project is an end-to-end system that:
- Classifies news articles as **Fake** or **Real** (ML).
- Generates a **short summary** of the article (DL Transformers).
- Exposes an **API** (FastAPI) and a simple **UI** (Streamlit).

---

## 📅 4-Week Execution Plan

### Week 1 — Data & Baseline
- [ ] Set up environment (`python -m venv venv` then `pip install -r requirements.txt`).
- [ ] Acquire dataset (e.g., Kaggle Fake News; columns like `text`, `label`).
- [ ] Clean & explore data (missing values, duplicates, label balance).
- [ ] Build **baseline TF‑IDF + Logistic Regression** classifier.
- [ ] Evaluate with Accuracy, F1, ROC-AUC. Save model to `models/`.

**Milestones:**
- `train_baseline.py` trains and saves `models/baseline.joblib`
- `evaluate.py` prints metrics on the test set

### Week 2 — Summarization & Pipeline
- [ ] Add summarization using **Hugging Face** (`BART`/`T5` pre-trained models).
- [ ] Implement chunking for long texts.
- [ ] Create a single **analyze()** pipeline: extract → classify → summarize.
- [ ] Add simple **URL scraper** to fetch article text from a link.

**Milestones:**
- `summarizer.py` with `summarize_text()`
- `scraper.py` with `get_article_text(url)`
- `predict.py` exposing `analyze(text_or_url)`

### Week 3 — API & UI
- [ ] Build **FastAPI** endpoint `/analyze` (accepts `url` or `text`).
- [ ] Build **Streamlit** UI: user pastes URL or text; show label, prob, summary.
- [ ] Add simple error handling & logging.

**Milestones:**
- `app/api.py` runs `uvicorn` server
- `app/streamlit_app.py` UI for local demo

### Week 4 — Polish & Deploy
- [ ] Add model monitoring metrics (latency, confidence).
- [ ] Improve classifier (try SVM, LinearSVC + Platt scaling, LightGBM).
- [ ] Optional: fine-tune summarizer on domain-specific data.
- [ ] Containerize with Docker; deploy backend (Railway/Render/AWS) and frontend (Vercel/Netlify).

**Milestones:**
- `Dockerfile` (optional)
- Deployed endpoints & UI

---

## 🏃 Quickstart (Local)

```bash
# 1) Create env and install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Put your dataset at data/fake_news.csv with columns: text,label
#    Labels: 1 for FAKE, 0 for REAL (or adapt in train script)

# 3) Train baseline
python src/train_baseline.py --data_path data/fake_news.csv --model_path models/baseline.joblib

# 4) Evaluate
python src/evaluate.py --data_path data/fake_news.csv --model_path models/baseline.joblib

# 5) Test summarize + classify quickly
python src/predict.py --text "Shocking! Aliens landed in Delhi yesterday..."

# 6) Run API
uvicorn src.app.api:app --reload --port 8000

# 7) Run UI
streamlit run src/app/streamlit_app.py
```

---

## 📁 Project Structure

```
fake_news_summarizer_project/
├─ data/
│  └─ fake_news.csv              # (you add this)
├─ models/
│  └─ baseline.joblib            # (generated after training)
├─ src/
│  ├─ preprocess.py
│  ├─ train_baseline.py
│  ├─ evaluate.py
│  ├─ predict.py
│  ├─ scraper.py
│  ├─ summarizer.py
│  └─ app/
│     ├─ api.py
│     └─ streamlit_app.py
├─ requirements.txt
└─ README.md
```

---

## ✅ Notes
- The baseline uses **TF‑IDF + Logistic Regression** with built-in English stopwords.
- Summarization uses **Transformers**; first run will download model weights.
- For better accuracy, consider cleaning sources, balancing labels, and trying multiple models.
