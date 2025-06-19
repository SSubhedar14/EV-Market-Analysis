# Tab1 - Sentiment Analysis Tab1
# This tab handles sentiment analysis of EV reviews, including visualizations and interactions.
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.sentiment import predict_sentiment_label, sentiment_breakdown
from utils.visualization import (
    plot_sentiment_pie,
    plot_wordcloud,
    plot_sentiment_bar,
    plot_sentiment_scatter
)

def render(data_2w, data_4w_cw, data_4w_cd, model, vectorizer):
    st.title("📊 Sentiment Analysis of EV Reviews")

    # Combine and normalize rating columns
    def normalize_data(df, vehicle_type):
        df = df.dropna(subset=["Review"]).copy()
        df["Vehicle_Type"] = vehicle_type
        if "rating" not in df.columns and "Rating" in df.columns:
            df.rename(columns={"Rating": "rating"}, inplace=True)
        elif "rating" not in df.columns:
            df["rating"] = np.nan  # Default if missing
        return df

    data_2w = normalize_data(data_2w, "2-Wheeler")
    data_4w_cw = normalize_data(data_4w_cw, "4-Wheeler")
    data_4w_cd = normalize_data(data_4w_cd, "4-Wheeler")

    all_data = pd.concat([data_2w, data_4w_cw, data_4w_cd], ignore_index=True)

    # Vehicle type selector
    vehicle_type = st.selectbox("Choose Vehicle Type", ["2-Wheeler", "4-Wheeler"])

    filtered_by_type = all_data[all_data["Vehicle_Type"] == vehicle_type]

    # Model selector
    selected_model = st.selectbox("Choose EV Model (or All)", ["All"] + sorted(filtered_by_type["Model_Name"].unique()))

    if selected_model != "All":
        filtered_data = filtered_by_type[filtered_by_type["Model_Name"] == selected_model].copy()
    else:
        filtered_data = filtered_by_type.copy()

    # Predict sentiment if missing
    if "Predicted Sentiment" not in filtered_data.columns:
        with st.spinner("Analyzing sentiment..."):
            filtered_data["Predicted Sentiment"] = filtered_data["Review"].apply(predict_sentiment_label)

    # Sentiment filter
    st.subheader("Sentiment Distribution")
    pie_fig = plot_sentiment_pie(filtered_data)
    st.plotly_chart(pie_fig, use_container_width=True)

    sentiments = ["All", "Positive", "Neutral", "Negative"]
    clicked_sentiment = st.radio("Filter by sentiment", sentiments, horizontal=True)

    if clicked_sentiment != "All":
        filtered_data = filtered_data[filtered_data["Predicted Sentiment"] == clicked_sentiment]
        st.success(f"Showing only **{clicked_sentiment}** reviews")

    # Word Cloud
    st.subheader("Most Frequent Words")
    wc_fig = plot_wordcloud(filtered_data, sentiment=clicked_sentiment if clicked_sentiment != "All" else None)
    st.pyplot(wc_fig)

    # Bar Chart
    st.subheader("Sentiment Count")
    bar_fig = plot_sentiment_bar(filtered_data, filtered_sentiment=clicked_sentiment if clicked_sentiment != "All" else None)
    st.plotly_chart(bar_fig, use_container_width=True)

    # Scatter Plot
    st.subheader("Review Length vs Sentiment")
    scatter_fig = plot_sentiment_scatter(filtered_data, filtered_sentiment=clicked_sentiment if clicked_sentiment != "All" else None)
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Heatmap
    # st.subheader("Ratings per Sentiment (Strip Plot)")
    # stripplot_fig = plot_rating_stripplot(filtered_data)

    # if stripplot_fig:
    #     st.pyplot(stripplot_fig)
    # else:
    #     st.info("Rating data not available for this selection.")
    
    # Summary Stats
    st.subheader("Review Summary Stats")
    total_reviews = len(filtered_data)
    avg_rating = filtered_data["rating"].mean() if "rating" in filtered_data.columns else "N/A"
    avg_length = filtered_data["Review"].astype(str).apply(len).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("Average Rating", f"{avg_rating:.2f}" if isinstance(avg_rating, float) else "N/A")
    col3.metric("Avg Review Length", f"{avg_length:.1f} characters")
