# app.py
import re
import numpy as np
import streamlit as st
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- Custom classes ----------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.url_re = re.compile(r'(https?://\S+|www\.\S+)')
        self.num_re = re.compile(r'\d+')
        self.phone_re = re.compile(r'\b\d{6,}\b')
    def fit(self, X, y=None): return self
    def transform(self, X):
        def clean(text):
            if X is None: return ""
            s = str(text).lower()
            s = self.url_re.sub(' URL ', s)
            s = self.phone_re.sub(' PHONENUM ', s)
            s = self.num_re.sub(' NUM ', s)
            s = re.sub(r'[^a-z0-9\s]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s
        return np.array([clean(t) for t in X]).reshape(-1,1)

class TextSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.ravel()

class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = []
        for t in X:
            s = str(t)
            length = len(s)
            digits = sum(c.isdigit() for c in s)
            uppercase_ratio = sum(1 for c in s if c.isupper())/max(1,len(s))
            has_url = 1 if 'url' in s else 0
            has_phone = 1 if 'phonenumber' in s or 'phonenum' in s else 0
            out.append([length, digits, uppercase_ratio, has_url, has_phone])
        return np.array(out)

# ⚡ Must be first Streamlit command after imports
st.set_page_config(page_title="SMS Spam Detector", layout="centered")

# ---------- Load pipeline ----------
@st.cache_resource
def load_pipeline(path='sms_spam_pipeline.joblib'):
    return joblib.load(path)

pipeline = load_pipeline()

# ---------- Page title ----------
st.title("SMS Spam Detection")
st.write("Enter an SMS message or upload CSV (column: text).")

# ---------- Single input ----------
text = st.text_area("Message", height=120)
if st.button("Predict Single"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        pred = pipeline.predict([text])[0]
        prob = pipeline.predict_proba([text])[0]
        idx = list(pipeline.classes_).index(pred)
        st.write(f"**Prediction:** {pred.upper()}")
        st.write(f"**Confidence:** {prob[idx]:.3f}")

# ---------- Batch upload ----------
uploaded = st.file_uploader("Upload CSV (column 'text')", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        if st.button("Predict Batch"):
            preds = pipeline.predict(df['text'].astype(str).tolist())
            probs = pipeline.predict_proba(df['text'].astype(str).tolist())
            df['prediction'] = preds
            df['confidence'] = probs.max(axis=1)
            st.write(df.head(50))
            st.download_button(
                "Download predictions CSV", 
                df.to_csv(index=False), 
                file_name='predictions.csv'
            )
