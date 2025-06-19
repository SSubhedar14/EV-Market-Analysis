import pandas as pd
from utils.sentiment import predict_numerical_score

def rank_models_by_preferences(df, preferences):
    attribute_cols = list(preferences.keys())
    model_group = df.groupby("Model_Name")[attribute_cols].mean().reset_index()

    # Normalize attributes 0-1
    for col in attribute_cols:
        min_val, max_val = model_group[col].min(), model_group[col].max()
        if max_val != min_val:
            model_group[col] = (model_group[col] - min_val) / (max_val - min_val)
        else:
            model_group[col] = 0.5

    total_weight = sum(preferences.values())
    model_group["Overall Score"] = sum(
        model_group[attr] * weight for attr, weight in preferences.items()
    ) / total_weight

    return model_group.sort_values("Overall Score", ascending=False).reset_index(drop=True)

def rank_models_by_textual_preferences(df, text_pref, attribute_cols, top_n=3):
    # Simple keyword to attribute mapping 
    keyword_map = {
        "performance": "Performance",
        "comfort": "Comfort",
        "value": "Value for Money",
        "mileage": "Mileage",
        "design": "Design",
        "battery": "Battery Life",
        
    }

    # Convert input to lowercase, find matching attributes
    text_lower = text_pref.lower()
    selected_attrs = [attr for kw, attr in keyword_map.items() if kw in text_lower and attr in attribute_cols]

    if not selected_attrs:
        # fallback: use all attributes equally weighted
        selected_attrs = attribute_cols

    # Equal weights if none specified
    preferences = {attr: 1 for attr in selected_attrs}

    ranked = rank_models_by_preferences(df, preferences)

    # Add sentiment scores averaged per model
    sentiment_scores = df.groupby("Model_Name")["Review"].apply(
        lambda reviews: reviews.apply(predict_numerical_score).mean()
    ).reset_index().rename(columns={"Review": "Sentiment Score"})

    # Merge sentiment with rankings
    ranked = ranked.merge(sentiment_scores, on="Model_Name", how="left")
    ranked["Sentiment Score"] = ranked["Sentiment Score"].fillna(ranked["Sentiment Score"].mean())

    # Combine overall score and sentiment score with weights (example weights)
    ranked["Combined Score"] = (ranked["Overall Score"] * 0.7) + (ranked["Sentiment Score"] / 5 * 0.3)

    ranked = ranked.sort_values("Combined Score", ascending=False).reset_index(drop=True)

    return ranked.head(top_n)
