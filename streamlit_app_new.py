#Use this code base only to run the app.py in local and then run this file if you waant to test this it is hosted in in streamlitcloud you can find the link in the repo.
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🧭", layout="centered")

def get_api_base_url_from_secrets():
    try:
        v = st.secrets["API_URL"]
        if v:
            return v.rstrip("/")
    except Exception:
        return ""

def get_api_token_from_secrets():
    try:
        return st.secrets["API_TOKEN"]
    except Exception:
        return None

api_url_secret = get_api_base_url_from_secrets()
api_token_secret = get_api_token_from_secrets()

st.title("SHL Assessment Recommender")
st.write("Paste a job description or natural language query and get recommended SHL assessments.")

api_url = st.text_input(
    "Recommendation API base URL (example: https://xxxx.ngrok-free.app)",
    value=api_url_secret,
    help="Enter the base URL only (do not include /recommend). If you set the secret API_URL it will be prefilled."
)

top_k = st.slider("Top K recommendations", min_value=1, max_value=10, value=7)

col1, col2 = st.columns([3,1])
with col1:
    query = st.text_area("Job description / Query", height=200)
with col2:
    uploaded = st.file_uploader("Or upload a .txt JD file", type=["txt"])
    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8")
            st.write("Uploaded JD preview:")
            st.write(content[:800])
            if not query.strip():
                query = content
        except:
            st.error("Failed to read file. Make sure it's a UTF-8 encoded text file.")

def call_recommend_api(base_url, q, k, token=None):
    endpoint = base_url.rstrip("/") + "/recommend"
    payload = {"query": q, "top_k": k}
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

if st.button("Get Recommendations"):
    if not api_url:
        st.error("Please enter the API base URL (ngrok or host) or set API_URL in Streamlit secrets.")
    elif not query or not query.strip():
        st.error("Please paste a job description or upload a JD file.")
    else:
        try:
            with st.spinner("Querying recommendation API..."):
                data = call_recommend_api(api_url, query, top_k, api_token_secret)
            recs = data.get("recommendations", [])
            if not recs:
                st.warning("No recommendations returned.")
            else:
                df = pd.DataFrame([{
                    "Assessment name": r.get("assessment_name", ""),
                    "Assessment URL": r.get("assessment_url", ""),
                    "Score": r.get("score", 0)
                } for r in recs])
                df["Assessment URL"] = df["Assessment URL"].astype(str)
                st.write("### Recommendations")
                st.dataframe(df[["Assessment name","Assessment URL","Score"]], use_container_width=True)

                # UI CSV (human-readable)
                csv_ui = df[["Assessment name", "Assessment URL", "Score"]].to_csv(index=False)
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
                csv_submission = submission_df.to_csv(index=False)
                st.download_button(
                    "📄 Download SHL Submission CSV (for evaluation)",
                    data=csv_submission,
                    file_name="submission_ready.csv",
                    mime="text/csv"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

st.write("---")
st.write("Tips:")
st.write("- If you deploy this on Streamlit Cloud, set API_URL in Secrets (recommended).")
st.write("- Make sure the Flask API is running and reachable at `/recommend`.")
