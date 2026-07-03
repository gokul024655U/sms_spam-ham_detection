# SMS Spam Detection Using Machine Learning

## Overview
This project is a Machine Learning-based SMS Spam Detection system that classifies SMS messages as Spam or Ham (Not Spam). The application uses TF-IDF feature extraction and Logistic Regression for classification. A Streamlit web application is provided for real-time message prediction.

## Features
- Single SMS prediction
- Batch prediction using CSV file
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression model
- Streamlit web interface
- Confidence score for predictions

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib

## Project Structure
```
app.py
train_model.py
project_spam.py
spam.csv
sms_spam_pipeline.joblib
requirements.txt
```

## Installation

1. Clone the repository
2. Install the required packages

```bash
pip install -r requirements.txt
```

## Run the Project

Train the model:

```bash
python train_model.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Dataset
- spam.csv

## Output
The application predicts whether an SMS message is:
- Spam
- Ham

It also displays the confidence score for each prediction.

## Author
**M. Gokulkumar**
