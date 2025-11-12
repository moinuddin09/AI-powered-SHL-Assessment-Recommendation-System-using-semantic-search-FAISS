# 🧭 SHL Assessment Recommendation System

An AI-powered recommendation engine that suggests the most relevant **SHL assessments** for any given **job description (JD)** or **natural language query**.  
The system leverages **semantic search using embeddings** and **keyword-aware re-ranking** to ensure accuracy and balanced results across technical and behavioral assessments.

---

## 🚀 Features
✅ Intelligent retrieval of assessments from SHL’s product catalog  
✅ Accepts natural language queries or full job descriptions  
✅ Dual output:
- Recruiter-friendly CSV (with scores)
- SHL submission-ready CSV (for evaluation)
✅ REST API built with Flask (`/health`, `/recommend`)  
✅ Streamlit web interface for interactive use  

---

## 🧠 Architecture Overview
**Core Components**
1. **Data ingestion** — SHL catalog pages scraped and augmented with provided dataset.  
2. **Embedding generation** — using `SentenceTransformer (all-MiniLM-L6-v2)`.  
3. **Indexing** — FAISS-based vector index for fast semantic retrieval.  
4. **Re-ranking** — combines semantic similarity and keyword overlap.  
5. **API layer** — Flask REST endpoints for recommendations.  
6. **UI layer** — Streamlit frontend for recruiters or hiring managers.

---

## ⚙️ Technology Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| Language | Python 3.10+ |
| Backend | Flask |
| Frontend | Streamlit |
| Model | SentenceTransformer (MiniLM-L6-v2) |
| Indexing | FAISS (CPU) |
| Data Processing | Pandas, NumPy, BeautifulSoup |
| Deployment | Localhost / ngrok / Render / Railway |
| Evaluation | Mean Recall@10 |

---

## 🧩 Folder Structure
```
SHL-Assessment-Recommender/
│
├── app.py                       # Flask API
├── streamlit_app.py              # Streamlit frontend
├── df_products.pkl               # Product data (pickle)
├── embeddings.npy                # Sentence embeddings
├── shl_catalog_augmented.csv     # Combined scraped + labeled data
├── recommendations_final.csv     # SHL submission format
├── SHL_Assessment_Approach.pdf   # 2-page approach document
├── requirements.txt
├── README.md                     # This file
└── .gitignore
```

---

## 🧰 Installation & Setup (Local with Conda)

### 1️⃣ Create environment
```bash
conda create -n shlapi python=3.10 -y
conda activate shlapi
```

### 2️⃣ Install dependencies
```bash
conda install -c pytorch faiss-cpu -y
pip install -r requirements.txt
```

### 3️⃣ Start the Flask API
```bash
python app.py
```
You’ll see:
```
 * Running on http://127.0.0.1:5000
```

### 4️⃣ Test the API
```bash
curl http://127.0.0.1:5000/health
```
→ `{"status": "ok"}`

```bash
curl -X POST http://127.0.0.1:5000/recommend      -H "Content-Type: application/json"      -d "{"query":"Python developer with SQL and JavaScript skills","top_k":5}"
```

---

## 🎨 Run the Streamlit Web App
```bash
streamlit run streamlit_app.py
```

### Then:
1. Paste your API base URL (`http://127.0.0.1:5000` or your ngrok URL)
2. Paste or upload a job description
3. Get recommendations instantly

You’ll get two download buttons:
- 📁 `recommendations_ui.csv` → Recruiter-friendly format  
- 📄 `submission_ready.csv` → SHL evaluation format  

---

## 🔗 API Endpoints

### `/health`
**GET** — simple check  
Response:  
```json
{ "status": "ok" }
```

### `/recommend`
**POST** — generate recommendations  
Example request:
```json
{
  "query": "Looking to hire Python developer with SQL and JavaScript skills",
  "top_k": 10
}
```

Example response:
```json
{
  "query": "...",
  "recommendations": [
    {
      "assessment_name": "Python (New)",
      "assessment_url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
      "score": 0.82
    }
  ]
}
```

---

## 📊 Evaluation Metric

**Mean Recall@10** — Measures how many relevant assessments appear in the top 10.  

```python
def mean_recall_at_k(gt_df, pred_df, K=10):
    queries = sorted(gt_df['Query'].unique())
    recalls = []
    for q in queries:
        gt_set = set(gt_df[gt_df['Query']==q]['Assessment_url'])
        preds = pred_df[pred_df['Query']==q]['Assessment_url'][:K]
        recalls.append(len(set(preds)&gt_set)/max(1,len(gt_set)))
    return sum(recalls)/len(recalls)
```

---

## 📁 Submission Package

| File | Purpose |
|------|----------|
| **recommendations_final.csv** | Predictions on test queries |
| **SHL_Assessment_Approach.pdf** | 2-page approach document |
| **GitHub URL** | Source code and experiments |
| **API Endpoint URL** | Deployed API link |
| **Streamlit URL** | Frontend to test interactively |

---

## 🧠 Author
**Name:** Moinuddin Navalur  
**Date:** November 2025  
**Email:** moinuddinnavalur6@gmail.com  
**Location:** Hubli, Karnataka, India  

---

## 🏁 License
MIT License — Free for academic and research use.

---

## ⭐ Acknowledgment
This project was developed as part of the **SHL Research Generative AI Assignment**, demonstrating how AI-driven recommendation systems can enhance assessment selection using semantic and keyword-based intelligence.
