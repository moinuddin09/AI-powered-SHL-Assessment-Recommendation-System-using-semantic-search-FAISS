import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import logging
import traceback
import pickle
from typing import Dict, Any, List

# --- Configuration and Setup ---
# We no longer need to check for environment variables for paths, 
# but we keep the logger setup.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-recommender")

MODEL_NAME = "all-MiniLM-L6-v2"
DF_PRODUCTS_PATH = "df_products.pkl"
EMBEDDINGS_PATH = "embeddings.npy"

# This list of tokens is pulled directly from your app.py
TECH_TOKENS = ["python", "sql", "javascript", "java", "excel", "tableau", "selenium", "automation", "react", "node", "css", "html"]


# --- 1. CORE LOGIC FUNCTIONS (Transferred from app.py) ---

def safe_load_pickle(path):
    """Safely loads a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

# Use Streamlit's caching instead of functools.lru_cache
@st.cache_resource(show_spinner=True)
def load_artifacts() -> Dict[str, Any]:
    """
    Loads all heavy artifacts (DataFrame, Embeddings, FAISS Index, Model) 
    and caches them for efficiency.
    """
    logger.info("Attempting to load models and data...")
    if not os.path.exists(DF_PRODUCTS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        st.error(f"Required artifacts missing. Check for {DF_PRODUCTS_PATH} and {EMBEDDINGS_PATH}.")
        raise FileNotFoundError("Required artifacts missing.")
    
    try:
        df_products = safe_load_pickle(DF_PRODUCTS_PATH)
        embeddings = np.load(EMBEDDINGS_PATH).astype('float32') # Ensure float32 for FAISS
        
        if embeddings.ndim != 2:
            raise ValueError("embeddings.npy must be a 2D numpy array")
        
        emb_dim = embeddings.shape[1]
        
        # Normalize and build FAISS Index (IndexFlatIP for dot product/cosine similarity)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(emb_dim)
        index.add(embeddings)
        
        model = SentenceTransformer(MODEL_NAME)
        
        st.success("Models and data loaded successfully!")
        return {
            "df_products": df_products,
            "embeddings": embeddings,
            "index": index,
            "model": model
        }
    except Exception as e:
        logger.error(f"Artifact loading failed: {e}")
        st.error(f"Failed to load required files or models: {e}")
        st.error(traceback.format_exc())
        return {}


def query_tokens(query: str) -> List[str]:
    """Identifies tech tokens present in the query."""
    q = query.lower()
    return [t for t in TECH_TOKENS if (" " + t + " ") in (" " + q + " ") or q.find(t) != -1]


def build_candidate_list(query: str, artifacts: Dict[str, Any], candidate_pool: int = 200) -> List[Dict[str, Any]]:
    """Performs semantic search and preliminary scoring."""
    model = artifacts["model"]
    index = artifacts["index"]
    df_products = artifacts["df_products"]
    
    # 1. Encode query
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    # 2. Search FAISS
    D, I = index.search(q_emb, min(candidate_pool, index.ntotal))
    
    sims = D[0].tolist()
    idxs = I[0].tolist()
    candidates = []
    
    for sim, idx in zip(sims, idxs):
        if idx < 0:
            continue
        
        prod = df_products.iloc[idx]
        
        # CORRECTED: Use the column names you provided
        txt = prod.get("__search_text__", (prod.get("name","") + " " + prod.get("desc","") + " " + prod.get("text",""))).lower()
        name = prod.get("name","")
        url = prod.get("url","")
        
        # Apply token score logic from your app.py
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


def token_strict_recommend_internal(query: str, artifacts: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Applies strict token filtering and final scoring, 
    returning results ready for the Streamlit UI.
    """
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
            
            # Penalize non-token matches if strict pool is small
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
            # The keys used here MUST match what the Streamlit UI expects
            "assessment_name": r["name"],
            "assessment_url": r["url"],
            "score": r["combined"]
        })
    return results


# --- 2. HELPER FUNCTION FOR CSV DOWNLOAD ---
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')


# --- 3. STREAMLIT UI CODE ---
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🧭", layout="centered")

st.title("SHL Assessment Recommender")
st.write("Paste a job description or natural language query and get recommended SHL assessments.")

# Load artifacts once at the start
artifacts = load_artifacts()

# Check if artifacts failed to load
if not artifacts:
    st.error("The application cannot run because the necessary models and data failed to load. Please fix the file path or content errors.")
else:
    # --- Input Fields ---
    top_k = st.slider("Top K recommendations", min_value=1, max_value=10, value=7)
    
    query_container = st.empty()
    file_container = st.empty()

    with query_container.container():
        query = st.text_area("Job description / Query", height=200, key="query_input")

    with file_container.container():
        uploaded = st.file_uploader("Or upload a .txt JD file", type=["txt"], key="file_upload")
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8")
                st.write("Uploaded JD preview:")
                st.write(content[:800])
                # If file is uploaded, use its content for the query
                if not query.strip():
                    query = content
            except Exception:
                st.error("Failed to read file. Make sure it's a UTF-8 encoded text file.")

    # --- Recommendations Button ---
    if st.button("Get Recommendations"):
        if not query or not query.strip():
            st.error("Please paste a job description or upload a JD file.")
        else:
            with st.spinner("Finding recommendations..."):
                try:
                    # DIRECT CALL TO THE INTERNAL LOGIC
                    recs = token_strict_recommend_internal(query, artifacts, top_k=top_k)
                    
                    if not recs:
                        st.warning("No recommendations returned.")
                    else:
                        # 4. Process results and create DataFrame
                        df = pd.DataFrame([{
                            "Assessment name": r.get("assessment_name", ""),
                            "Assessment URL": r.get("assessment_url", ""),
                            "Score": r.get("score", 0)
                        } for r in recs])
                        
                        df["Assessment URL"] = df["Assessment URL"].astype(str)
                        st.write("### Recommendations")
                        
                        # Display DataFrame
                        st.dataframe(
                            df[["Assessment name","Assessment URL","Score"]].style.format({"Score": "{:.4f}"}), 
                            use_container_width=True
                        )

                        # 5. Download Buttons
                        # UI CSV (human-readable)
                        csv_ui = convert_df_to_csv(df[["Assessment name", "Assessment URL", "Score"]])
                        st.download_button(
                            "📁 Download UI CSV (Human-readable)",
                            data=csv_ui,
                            file_name="recommendations_ui.csv",
                            mime="text/csv"
                        )

                        # SHL Submission-ready CSV
                        submission_df = pd.DataFrame({
                            "Query": [query] * len(df),
                            "Assessment_url": df["Assessment URL"]
                        })
                        csv_submission = convert_df_to_csv(submission_df)
                        st.download_button(
                            "📄 Download SHL Submission CSV (for evaluation)",
                            data=csv_submission,
                            file_name="submission_ready.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"An unexpected error occurred during recommendation:")
                    st.error(traceback.format_exc())
