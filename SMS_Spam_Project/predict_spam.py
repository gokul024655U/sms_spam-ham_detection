import re
import joblib
import numpy as np
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

# ---------- Load pipeline ----------
pipeline = joblib.load('sms_spam_pipeline.joblib')

# ---------- Test messages ----------
sms_samples = [
    "Congratulations! You've won a free iPhone. Click here to claim: http://spam.com",
    "Hey, are we still meeting for lunch today?"
]

predictions = pipeline.predict(sms_samples)

for sms, label in zip(sms_samples, predictions):
    print(f"SMS: {sms}\nPredicted label: {label}\n")
