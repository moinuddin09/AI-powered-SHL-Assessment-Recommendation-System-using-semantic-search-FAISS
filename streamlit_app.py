import streamlit as st
import pandas as pd
import numpy as np
import faiss  # For the vector search
from sentence_transformers import SentenceTransformer  # For loading the model
import io  # For CSV download
import traceback # For debugging errors

# --- 1. LOAD MODELS AND DATA (The "Brain" Setup) ---
# This function loads all your heavy models and data ONCE.
# @st.cache_resource tells Streamlit to store this in memory
# so it doesn't reload every time a user clicks a button.
@st.cache_resource
def load_models_and_data():
    """
    Loads the Sentence Transformer model, the product dataframe,
    and builds the FAISS index from the embeddings.
    """
    try:
        # Load the same model you used to create the embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the product data
        df = pd.read_pickle("df_products.pkl")
        
        # Load the embeddings
        embeddings = np.load("embeddings.npy")
        
        # --- Build the FAISS Index ---
        # Get the dimension of the embeddings (e.g., 384 for MiniLM)
        d = embeddings.shape[1]
        
        # We use IndexFlatIP, which is good for cosine similarity 
        # (what SBERT models use)
        index = faiss.IndexFlatIP(d)
        
        # IMPORTANT: Normalize the embeddings before adding to IndexFlatIP
        # This makes the "dot product" (IP) search equivalent to cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add the normalized embeddings to the index
        index.add(embeddings.astype('float32'))
        
        st.success("Models and data loaded successfully!")
        return model, df, index
        
    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}. Make sure 'df_products.pkl' and 'embeddings.npy' are in your GitHub repo.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.error(traceback.format_exc())
        return None, None, None

# --- 2. THE RECOMMENDATION FUNCTION (The "Brain" Logic) ---
# This function replaces your /recommend API endpoint.
def get_recommendations(query, index, model, df, top_k=10):
    """
    Takes a user query and returns two DataFrames:
    1. df_ui: For displaying in the Streamlit app (with scores)
    2. df_submission: For the SHL submission format
    """
    # 1. Encode the user's query into an embedding
    query_embedding = model.encode([query])
    
    # 2. Normalize the query embedding for cosine similarity search
    faiss.normalize_L2(query_embedding)
    
    # 3. Search the FAISS index
    # D = distances (scores), I = indices (row numbers in your df)
    D, I = index.search(query_embedding.astype('float32'), top_k)
    
    # 4. Get the results from the indices
    # I[0] contains the list of indices for our single query
    indices = I[0]
    
    # Filter out invalid indices (e.g., -1 if not enough results)
    valid_indices = [i for i in indices if i != -1]
    
    if not valid_indices:
        return pd.DataFrame(), pd.DataFrame() # Return empty dataframes

    # 5. Get the actual data from your dataframe
    recommendations = df.iloc[valid_indices].copy()
    
    # Add the scores (D[0] contains the scores)
    recommendations['score'] = D[0][0:len(valid_indices)]
    
    # --- IMPORTANT ---
    # I am GUESSING your column names are 'Assessment_name' and 'Assessment_url'
    # based on your README. If your .pkl file has different names,
    # you MUST change them in the two lines below.
    try:
        # 6. Format the UI DataFrame
        df_ui = recommendations[['Assessment_name', 'Assessment_url', 'score']]
        df_ui = df_ui.sort_values(by='score', ascending=False)
    
        # 7. Format the Submission DataFrame
        df_submission = pd.DataFrame()
        df_submission['Query'] = [query] * len(df_ui)
        df_submission['Assessment_url'] = df_ui['Assessment_url']
        
        return df_ui, df_submission
        
    except KeyError as e:
        st.error(f"Column not found: {e}. Please check your column names in `df_products.pkl` and update the code.")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. HELPER FUNCTION FOR CSV DOWNLOAD ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- 4. LOAD EVERYTHING (This runs only once) ---
# Load the models when the app starts.
model, df_products, faiss_index = load_models_and_data()


# --- 5. THE STREAMLIT APP (The "Face" / UI) ---
st.title("🚀 SHL Assessment Recommendation System")
st.markdown("Enter a job description or query to find the most relevant SHL assessments.")

# Check if models loaded correctly before showing the UI
if model and df_products is not None and faiss_index is not None:
    
    # Get user input
    user_query = st.text_area("Enter Job Description or Query", "Python developer with SQL and JavaScript skills", height=150)
    top_k = st.number_input("Number of recommendations (Top K)", min_value=1, max_value=50, value=10)

    # The "Get Recommendations" button
    if st.button("Get Recommendations"):
        if not user_query.strip():
            st.error("Please enter a job description or query.")
        else:
            with st.spinner("Finding recommendations... This may take a moment."):
                try:
                    # --- THIS IS THE KEY CHANGE ---
                    # We are no longer calling an API with `requests.post`
                    # We are calling our Python function directly!
                    df_ui, df_submission = get_recommendations(
                        query=user_query, 
                        index=faiss_index, 
                        model=model, 
                        df=df_products, 
                        top_k=int(top_k)
                    )
                    
                    if df_ui.empty:
                        st.warning("No recommendations found for that query.")
                    else:
                        st.subheader("Recommended Assessments")
                        st.dataframe(df_ui.style.format({"score": "{:.4f}"}))
                        
                        st.markdown("---")
                        st.subheader("Download Results")
                        
                        # --- Download Button for UI CSV ---
                        csv_ui = convert_df_to_csv(df_ui)
                        st.download_button(
                            label="Download Recruiter CSV (with scores)",
                            data=csv_ui,
                            file_name="recommendations_ui.csv",
                            mime="text/csv",
                        )
                        
                        # --- Download Button for Submission CSV ---
                        csv_submission = convert_df_to_csv(df_submission)
                        st.download_button(
                            label="Download Submission CSV (for SHL)",
                            data=csv_submission,
                            file_name="submission_ready.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"An error occurred while getting recommendations:")
                    st.error(e)
                    st.error(traceback.format_exc())
else:
    st.warning("Models could not be loaded. The app is non-functional. Please check the logs.")
