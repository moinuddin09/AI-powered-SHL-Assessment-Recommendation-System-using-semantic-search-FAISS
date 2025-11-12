import os
import logging
import traceback
from functools import lru_cache

from flask import Flask, request, jsonify
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DF_PRODUCTS_PATH = os.getenv("DF_PRODUCTS_PKL", "df_products.pkl")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_NPY", "embeddings.npy")

TECH_TOKENS = ["python", "sql", "javascript", "java", "excel", "tableau", "selenium", "automation", "react", "node", "css", "html"]

# ---------- Utilities and loading ----------
def safe_load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_artifacts():
    if not os.path.exists(DF_PRODUCTS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("Required artifacts missing. Place df_products.pkl and embeddings.npy next to app.py or set DF_PRODUCTS_PKL/EMBEDDINGS_NPY env vars.")
    df_products = safe_load_pickle(DF_PRODUCTS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy must be a 2D numpy array")
    emb_dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(emb_dim)
    index.add(embeddings)
    model = SentenceTransformer(MODEL_NAME)
    return {
        "df_products": df_products,
        "embeddings": embeddings,
        "index": index,
        "model": model
    }

def query_tokens(query):
    q = query.lower()
    return [t for t in TECH_TOKENS if (" " + t + " ") in (" " + q + " ") or q.find(t) != -1]

def build_candidate_list(query, artifacts, candidate_pool=200):
    model = artifacts["model"]
    index = artifacts["index"]
    df_products = artifacts["df_products"]
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, min(candidate_pool, index.ntotal))
    sims = D[0].tolist()
    idxs = I[0].tolist()
    candidates = []
    for sim, idx in zip(sims, idxs):
        if idx < 0:
            continue
        prod = df_products.iloc[idx]
        txt = prod.get("__search_text__", (prod.get("name","") + " " + prod.get("desc","") + " " + prod.get("text",""))).lower()
        name = prod.get("name","")
        url = prod.get("url","")
        prod_toks = [t for t in TECH_TOKENS if (" " + t + " ") in (" " + txt + " ") or t in url.lower() or t in name.lower()]
        prod_token_score = min(1.0, len(prod_toks) / 2.0)
        combined = float(sim) + 0.45 * prod_token_score
        candidates.append({
            "idx": int(idx),
            "name": name,
            "url": url,
            "sim": float(sim),
            "prod_toks": prod_toks,
            "token_score": prod_token_score,
            "combined": combined
        })
    return candidates

def token_strict_recommend_internal(query, artifacts, top_k=10):
    q = query if isinstance(query, str) else str(query)
    q_tokens = query_tokens(q)
    cands = build_candidate_list(q, artifacts, candidate_pool=200)
    if len(q_tokens) > 0:
        strict = [c for c in cands if any(qt in c["prod_toks"] for qt in q_tokens)]
        if len(strict) >= top_k:
            out = sorted(strict, key=lambda x: x["combined"], reverse=True)[:top_k]
        else:
            out = sorted(strict, key=lambda x: x["combined"], reverse=True)
            remaining = [c for c in cands if c not in out]
            for r in remaining:
                if r["token_score"] == 0.0:
                    r["combined"] = r["combined"] - 0.5
            remaining_sorted = sorted(remaining, key=lambda x: x["combined"], reverse=True)
            out.extend(remaining_sorted[: max(0, top_k - len(out)) ])
    else:
        out = sorted(cands, key=lambda x: x["combined"], reverse=True)[:top_k]
    results = []
    for r in out:
        results.append({
            "assessment_name": r["name"],
            "assessment_url": r["url"],
            "score": r["combined"]
        })
    return results

# ---------- Flask routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error":"Invalid JSON body"}), 400

        q = payload.get("query") or payload.get("text") or payload.get("url") or payload.get("job_description")
        top_k = int(payload.get("top_k", 10))

        if not q:
            return jsonify({"error":"Missing 'query' or 'text' in JSON body"}), 400

        if isinstance(q, str) and q.lower().startswith("http"):
            try:
                import requests
                from bs4 import BeautifulSoup
                resp = requests.get(q, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                q_text = soup.get_text(" ", strip=True)[:6000]
            except Exception:
                q_text = q
        else:
            q_text = q

        artifacts = load_artifacts()
        recs = token_strict_recommend_internal(q_text, artifacts, top_k=top_k)
        return jsonify({"query": q_text[:800], "recommendations": recs}), 200

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Error in /recommend handler")
        return jsonify({"error":"Internal server error", "traceback_last_lines": tb.splitlines()[-10:]}), 500

# ---------- startup ----------
if __name__ == "__main__":
    logger.info("Starting SHL recommender API")
    # Ensure artifacts exist before running
    try:
        load_artifacts()
    except Exception as e:
        logger.error("Artifact loading failed: %s", e)
        raise
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
xcept Exception as e:
        logger.error("Artifact loading failed: %s", e)
        raise
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
