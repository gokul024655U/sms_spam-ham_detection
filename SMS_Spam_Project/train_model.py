import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------- Preprocessing ----------
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.url_re = re.compile(r'(https?://\S+|www\.\S+)')
        self.num_re = re.compile(r'\d+')
        self.phone_re = re.compile(r'\b\d{6,}\b')
    def fit(self, X, y=None): return self
    def transform(self, X):
        def clean(text):
            if pd.isna(text): return ""
            s = str(text).lower()
            s = self.url_re.sub(' URL ', s)
            s = self.phone_re.sub(' PHONENUM ', s)
            s = self.num_re.sub(' NUM ', s)
            s = re.sub(r'[^a-z0-9\s]', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s
        return X.astype(str).apply(clean).values.reshape(-1, 1)

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

# ---------- Load Dataset ----------
def load_data(path='spam.csv'):
    df = pd.read_csv(path, encoding='latin-1', usecols=[0,1], names=['label','text'], header=0)
    df = df.dropna(subset=['text','label'])
    df['label'] = df['label'].map(lambda x: 'spam' if str(x).lower().startswith('s') else 'ham')
    return df

def main():
    df = load_data('spam.csv')
    X, y = df['text'], df['label']

    pre = TextPreprocessor()
    X_clean = pd.Series(pre.transform(X).ravel())

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, stratify=y, random_state=42)

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)

    text_pipe = Pipeline([('selector', TextSelector()), ('tfidf', tfidf)])
    feat_pipe = Pipeline([('selector', TextSelector()), ('hand', HandcraftedFeatures())])
    union = FeatureUnion([('tfidf', text_pipe), ('hand', feat_pipe)])

    pipeline = Pipeline([
        ('preproc', pre),
        ('union', union),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
    ])

    param_grid = {'clf__C': [0.1, 1.0, 5.0]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_macro', verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(best, 'sms_spam_pipeline.joblib')
    print("Saved pipeline to sms_spam_pipeline.joblib")

if __name__ == '__main__':
    main()
