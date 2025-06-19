# Comparision tab 2
import streamlit as st
import pandas as pd
from utils.sentiment import predict_numerical_score
from utils.visualization import plot_sentiment_comparison_bar

def render(data_2w, data_4w_cw, data_4w_cd):
    st.title("🔍 EV Model Comparison")

    # Step 1: Select vehicle category
    category = st.radio("Select Vehicle Type", ["2 Wheeler", "4 Wheeler"])

    if category == "2 Wheeler":
        comparison_data = data_2w.copy()
    else:
        comparison_data = pd.concat([data_4w_cw, data_4w_cd], ignore_index=True)

    comparison_data.dropna(subset=["Review", "Model_Name"], inplace=True)
    all_models = sorted(comparison_data["Model_Name"].unique())

    # Step 2: Select models to compare
    selected_models = st.multiselect("Select two or more models to compare", all_models)

    if len(selected_models) >= 2:
        st.subheader("🔍 Sentiment Score Comparison")

        # Step 3: Compute average sentiment scores
        sentiment_scores = {
            model: comparison_data[comparison_data["Model_Name"] == model]["Review"]
                    .apply(predict_numerical_score).mean()
            for model in selected_models
        }

        # Step 4: Plot the comparison
        fig_sentiment = plot_sentiment_comparison_bar(sentiment_scores)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    else:
        st.warning("Please select at least two models to compare.")
