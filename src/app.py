import streamlit as st
import pickle
import re
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="ðŸ’°",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #f8f9fa;
        color: #2d3436;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title styling */
    h1 {
        color: #2d3436;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Text input styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 15px;
        font-size: 16px;
    }
    
    .stTextArea textarea:focus {
        border-color: #6c5ce7;
        box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #6c5ce7;
        color: white;
        font-weight: 600;
        padding: 10px 30px;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #a29bfe;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
    }
    
    /* Result styling */
    .result-container {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fake-result {
        background-color: #ffeaa7;
        border-left: 10px solid #fab1a0;
        color: #d63031;
    }
    
    .real-result {
        background-color: #55efc4;
        border-left: 10px solid #00b894;
        color: #006266;
    }
    
    .status-badge {
        font-size: 1.5rem;
        font-weight: 900;
        display: block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    tfidf_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    svm_path = os.path.join('models', 'svm_model.pkl')
    
    if not os.path.exists(tfidf_path) or not os.path.exists(svm_path):
        st.error("Model files not found! Please run the training script first.")
        return None, None
        
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(svm_path, 'rb') as f:
        svm_model = pickle.load(f)
    return tfidf, svm_model

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- APP LAYOUT ---
def main():
    st.title("ðŸ—žï¸ Fake News Detector")
    st.markdown("Enter a news article below to verify its authenticity using our trained SVM classifier.")
    
    tfidf, svm_model = load_models()
    
    # Input Area
    news_text = st.text_area("Article Content", placeholder="Paste news content here...", height=250)
    
    if st.button("Analyze Authenticity"):
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
        elif tfidf is None:
            st.error("System error: Models could not be loaded.")
        else:
            with st.spinner("Analyzing text patterns..."):
                # Preprocess
                cleaned = clean_text(news_text)
                vectorized = tfidf.transform([cleaned])
                
                # Predict
                prediction = svm_model.predict(vectorized)[0]
                
                # Display Results
                if prediction == 'FAKE':
                    st.markdown(f"""
                    <div class="result-container fake-result">
                        <span class="status-badge">âš ï¸ POTENTIALLY FAKE</span>
                        This article shows characteristics of misinformation. Proceed with caution.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-container real-result">
                        <span class="status-badge">âœ… LIKELY REAL</span>
                        This article appears to be authentic based on its linguistic patterns.
                    </div>
                    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption("Powered by Scikit-learn LinearSVC and TF-IDF Feature Extraction.")

if __name__ == "__main__":
    main()
