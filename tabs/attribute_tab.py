# attribute_tab.py tab3
import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.visualization import plot_attribute_score_analysis  # Assumed plotly version


def render(data_2w, data_4w_cw, data_4w_cd):
    st.title("🔧 Attribute Score Visualization")

    vehicle_type = st.selectbox("Select Vehicle Type", ["2 Wheeler", "4 Wheeler"])

    if vehicle_type == "2 Wheeler":
        combined = data_2w.copy()
        model_options = sorted(combined["Model_Name"].dropna().unique())
    else:
        # Combine both categories of 4 wheelers for data,
        # but only show model names from data_4w_cw in dropdown
        combined = pd.concat([data_4w_cw, data_4w_cd], ignore_index=True)
        model_options = sorted(data_4w_cw["Model_Name"].dropna().unique())

    if "Model_Name" not in combined.columns or "Review" not in combined.columns:
        st.error("Model_Name or Review column missing in dataset.")
        return

    def extract_attributes(df):
        drop_cols = [
            'Review', 'rating', 'Model_Name', 'Attributes Mentioned', 'Used it for', 'Owned for',
            'Ridden for', 'driven', 'Condition', 'Experience', 'Extra Features', 'Maintenance cost'
        ]
        return [col for col in df.columns if col not in drop_cols]

    attribute_cols = extract_attributes(combined)

    if not attribute_cols:
        st.warning("No attribute columns found in dataset.")
        return

    selected_model = st.selectbox("Select EV Model", model_options)

    model_data = combined[combined["Model_Name"] == selected_model]

    if model_data.empty:
        st.warning("No data found for this model.")
        return

    attributes_df = model_data.reindex(columns=attribute_cols).apply(pd.to_numeric, errors='coerce')
    avg_scores = attributes_df.mean()

    review_count = len(model_data)
    avg_length = model_data["Review"].astype(str).apply(len).mean()

    st.markdown(f"**Model:** `{selected_model}` | **Reviews:** `{review_count}` | **Avg. Review Length:** `{avg_length:.1f}` characters")

    fig = plot_attribute_score_analysis(selected_model, avg_scores, review_count, avg_length)
    st.plotly_chart(fig, use_container_width=True)