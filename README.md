# EV Market Analysis 🚗⚡

This is a Dissertation Project titled **"Leveraging Machine Learning and NLP for Electric Vehicle Market Analysis and Product Launch Success"**. It combines exploratory data analysis (EDA), natural language processing (NLP), and machine learning (ML) models to gain insights from EV reviews and forecast market trends.

## 🔍 Key Features

- Exploratory data analysis for two-wheeler and four-wheeler EV reviews
- Sentiment analysis using NLP (TF-IDF + sentiment scores)
- Predictive modeling using ML algorithms (Logistic Regression, AdaBoost, Random Forest, Ridge Classifier, LightGBM)
- Streamlit-powered interactive web application with using LightGBM Model (Highest Accueacy):
  - 1. Sentiment Analysis: 4 interactive graphs that support Cross Filtering and Cross Highlighting.
  - 2. Market Comparison: Compares sentiment scores of different EV models.
  - 3. Attribute-Based Analysis: Analyzes attribute scores for selected EV model.
  - 4. Best EV Recommendation: Recommends the top EV models based on user-selected/ described preferences.

## 🗂️ Project Structure
```
EV-Market-Analysis/
│
├── data/
│ ├── 2-wheeler-EV-bikewale.csv
│ ├── 4-wheeler-EV-carwale.csv
│ └── 4-wheeler-EV-cardekho.csv
│
├── models/
│ ├── lightgbm_model.pkl
│ └── vectorizer.pkl
│
├── tabs/
│ ├── attribute_tab.py
│ ├── comparison_tab.py
│ ├── recommendation_tab.py
│ └── sentiment_tab.py
│
├── utils/
│ ├── __init__.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── recommendation.py
│ ├── sentiment.py
│ └── visualization.py
│
├── app.py
├── EV_Market_Analysis.ipynb
├── requirements.txt
└── README.md
```
