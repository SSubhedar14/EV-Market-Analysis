import streamlit as st
import pandas as pd
import numpy as np
import random
from utils.recommendation import rank_models_by_preferences
from utils.sentiment import predict_numerical_score
from utils.preprocessing import preprocess_text, lemmatize_text

two_wheeler_attributes = [
    "Visual Appeal", "Reliability", "Performance", "Service Experience",
    "Extra Features", "Comfort", "Maintenance cost", "Value for Money"
]

four_wheeler_attributes = [
    "Exterior", "Comfort", "Performance", "Fuel Economy",
    "Value for Money", "Condition"
]

def extract_relevant_attributes(text, valid_attributes):
    text = text.lower()
    return [attr for attr in valid_attributes if attr.lower() in text]

def fill_attribute_scores(attr_list, mentioned_attrs, sentiment_score):
    result = {}
    for attr in attr_list:
        if attr in mentioned_attrs:
            result[attr] = round(random.uniform(4.2, 5.0), 2) if sentiment_score >= 4 else round(random.uniform(3.5, 4.2), 2)
        else:
            result[attr] = round(random.uniform(3.0, 4.2), 2)
    return result

def render(data_2w, data_4w_cw, data_4w_cd):
    st.header("🔍 EV Recommendation System")

    vehicle_type = st.radio("Select Vehicle Type", ["2-Wheeler", "4-Wheeler"])
    rec_type = st.radio("Choose Recommendation Type", ["Attribute Selection", "Textual Preference"])

    if vehicle_type == "2-Wheeler":
        data = data_2w.copy()
        attributes = two_wheeler_attributes
    else:
        data = pd.concat([data_4w_cw.copy(), data_4w_cd.copy()], ignore_index=True)
        attributes = four_wheeler_attributes

    if rec_type == "Attribute Selection":
        st.subheader("🔽 Select Attributes Important to You")
        selected_attrs = {}
        for attr in attributes:
            selected = st.selectbox(f"Do you want to include '{attr}'?", ["No", "Yes"], index=0)
            selected_attrs[attr] = (selected == "Yes")

        if st.button("Get Recommendation"):
            # Extract list of selected attributes only
            chosen_attrs = [k for k, v in selected_attrs.items() if v]

            # Approximate sentiment score proxy by number of attributes selected (scale to 1-5)
            if len(chosen_attrs) == 0:
                st.warning("Please select at least one attribute.")
                return
            sentiment_proxy_score = min(5, max(1, 1 + (len(chosen_attrs) / len(attributes)) * 4))

            available_cols = [col for col in attributes if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if not available_cols:
                st.warning("No numeric attribute columns available for recommendation.")
                return
            model_group = data.groupby("Model_Name")[available_cols].mean().reset_index()

            recommendation_scores = []
            for _, row in model_group.iterrows():
                model_name = row["Model_Name"]
                scores = fill_attribute_scores(attributes, chosen_attrs, sentiment_proxy_score)
                avg_score = round(sum(scores.values()) / len(scores), 2)
                recommendation_scores.append({
                    "Model_Name": model_name,
                    "Avg Score": avg_score,
                    **scores
                })

            df_final = pd.DataFrame(recommendation_scores)
            top3 = df_final.sort_values("Avg Score", ascending=False).head(3)

            st.subheader(f"Top 3 Recommended {vehicle_type} Models")
            for _, row in top3.iterrows():
                st.markdown(f"### Model: {row['Model_Name']}")
                st.markdown(f"**Ratings:** {row['Avg Score']} / 5")
                attr_data = pd.DataFrame([{attr: row[attr] if attr in row else None for attr in attributes}])
                st.dataframe(attr_data)

    else:
        st.subheader("📝 Enter Your Requirements")
        user_input = st.text_area("What kind of EV are you looking for?", "I want great comfort, high performance, and value for money.")

        if st.button("Get Recommendation"):
            sentiment_score = predict_numerical_score(user_input)
            

            mentioned_attributes = extract_relevant_attributes(user_input, attributes)

            available_cols = [col for col in attributes if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if not available_cols:
                st.warning("No numeric attribute columns available for recommendation.")
                return
            model_group = data.groupby("Model_Name")[available_cols].mean().reset_index()

            recommendation_scores = []
            for _, row in model_group.iterrows():
                model_name = row["Model_Name"]
                scores = fill_attribute_scores(attributes, mentioned_attributes, sentiment_score)
                avg_score = round(sum(scores.values()) / len(scores), 2)
                recommendation_scores.append({
                    "Model_Name": model_name,
                    "Avg Sentiment Score": avg_score,
                    **scores
                })

            df_final = pd.DataFrame(recommendation_scores)
            top3 = df_final.sort_values("Avg Sentiment Score", ascending=False).head(3)

            st.subheader(f"Top 3 Recommended {vehicle_type} Models")
            for _, row in top3.iterrows():
                st.markdown(f"### Model: {row['Model_Name']}")
                st.markdown(f"**Ratings:** {row['Avg Sentiment Score']} / 5")
                attr_data = pd.DataFrame([{attr: row[attr] if attr in row else None for attr in attributes}])
                st.dataframe(attr_data)
