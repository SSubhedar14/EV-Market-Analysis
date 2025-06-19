import streamlit as st
from utils.data_loader import load_data
from utils.sentiment import load_model_and_vectorizer
from tabs import sentiment_tab, comparison_tab, attribute_tab, recommendation_tab
import pandas as pd

def main():
    st.set_page_config(page_title="EV Review Sentiment Analysis", layout="wide")

    # Load datasets and models once
    data_2w, data_4w_cw, data_4w_cd = load_data()
    model, vectorizer = load_model_and_vectorizer()

    # ✅ Correct concatenation
    combined_data = pd.concat([data_2w, data_4w_cw, data_4w_cd], ignore_index=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    tabs = {
        "Sentiment Analysis": sentiment_tab,
        "Market Comparison": comparison_tab,
        "Attribute Scores": attribute_tab,
        "Recommendations": recommendation_tab
    }
    selected_tab = st.sidebar.radio("Go to", list(tabs.keys()))

    # Render the selected tab
    if selected_tab == "Sentiment Analysis":
        sentiment_tab.render(data_2w, data_4w_cw, data_4w_cd, model, vectorizer)
    elif selected_tab == "Market Comparison":
        comparison_tab.render(data_2w, data_4w_cw, data_4w_cd)
    elif selected_tab == "Attribute Scores":
        attribute_tab.render(data_2w, data_4w_cw, data_4w_cd)
    elif selected_tab == "Recommendations":
        recommendation_tab.render(data_2w, data_4w_cw, data_4w_cd)

if __name__ == "__main__":
    main()
