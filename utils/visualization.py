import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd



# Color palette for sentiment
sentiment_colors = {
    "Positive": "#66BB6A",
    "Neutral": "#42A5F5",
    "Negative": "#EF5350"
}

# PIE CHART
def plot_sentiment_pie(df):
    sentiment_counts = df["Predicted Sentiment"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        marker=dict(colors=[sentiment_colors.get(s, "#BDBDBD") for s in sentiment_counts.index]),
        hoverinfo="label+percent+value",
        textinfo="label+percent",
        pull=[0.1 if val == max(sentiment_counts.values) else 0 for val in sentiment_counts.values],
    )])
    fig.update_layout(
        title="Sentiment Distribution",
        clickmode='event+select'
    )
    return fig

# WORD CLOUD (filtered)
def plot_wordcloud(df, sentiment=None):
    if sentiment:
        df = df[df['Predicted Sentiment'] == sentiment]

    text = " ".join(df["Review"].astype(str).values)
    if not text.strip():
        text = "No words to show"

    wordcloud = WordCloud(
        width=800, height=400, background_color='white'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# BAR CHART
def plot_sentiment_bar(df, filtered_sentiment=None):
    if filtered_sentiment:
        df = df[df["Predicted Sentiment"] == filtered_sentiment]

    sentiment_counts = df["Predicted Sentiment"].value_counts()
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        color_discrete_map=sentiment_colors,
        labels={"x": "Sentiment", "y": "Count"},
        title="Sentiment Count Distribution"
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    fig.update_layout(clickmode='event+select')
    return fig

# SCATTER PLOT
def plot_sentiment_scatter(df, filtered_sentiment=None):
    df["Review Length"] = df["Review"].astype(str).apply(len)
    if filtered_sentiment:
        df = df[df["Predicted Sentiment"] == filtered_sentiment]

    fig = px.scatter(
        df,
        x="Review Length",
        y="Predicted Sentiment",
        color="Predicted Sentiment",
        color_discrete_map=sentiment_colors,
        hover_data=["Review"],
        title="Review Length vs Predicted Sentiment"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    return fig

# HEATMAP: Sentiment vs Rating
# def plot_sentiment_rating_heatmap(df, filtered_sentiment=None):
#     if filtered_sentiment:
#         df = df[df["Predicted Sentiment"] == filtered_sentiment]

#     if "rating" not in df.columns:
#         return None

#     heatmap_df = pd.crosstab(df["Predicted Sentiment"], df["rating"])
#     fig = go.Figure(data=go.Heatmap(
#         z=heatmap_df.values,
#         x=heatmap_df.columns.astype(str),
#         y=heatmap_df.index,
#         colorscale="Blues",
#         hovertemplate="Rating: %{x}<br>Sentiment: %{y}<br>Count: %{z}<extra></extra>"
#     ))
#     fig.update_layout(title="Sentiment vs Rating Heatmap")
#     return fig

import plotly.express as px
import pandas as pd

def plot_attribute_score_analysis(model_name, avg_scores, review_count, avg_length):
    # Drop unwanted attributes
    avg_scores = avg_scores.drop(labels=[col for col in ['rating', 'vehicle_type'] if col in avg_scores.index])

    # Create dataframe
    df = pd.DataFrame({
        "Attribute": avg_scores.index,
        "Avg Score": avg_scores.values
    })

    # Remove attributes with all-NaN values
    df = df.dropna()

    # Plotly bar chart
    fig = px.bar(
        df,
        x="Attribute",
        y="Avg Score",
        color="Attribute",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"🔧 Attribute Scores for {model_name}",
        labels={"Avg Score": "Average Score (1 to 5)"},
    )

    # Formatting the chart
    
    fig.update_layout(
        xaxis_title="Attributes",
        yaxis_title="Score (1-5)",
        yaxis=dict(tickmode='linear', tick0=1, dtick=1, range=[1, 5]),
        font=dict(color="black", size=14),
        title_font=dict(color="black", size=18),
        legend=dict(font=dict(color="black")),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=120)
    )

    fig.update_traces(marker_line_color='black', marker_line_width=1)

    return fig


def plot_sentiment_comparison_bar(sentiment_scores: dict):
    sentiment_df = pd.DataFrame({
        "Model": list(sentiment_scores.keys()),
        "Avg Sentiment Score": list(sentiment_scores.values())
    })

    fig = px.bar(
        sentiment_df,
        x="Model",
        y="Avg Sentiment Score",
        title="📊 Average Sentiment Score per Model",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Pastel1,
        labels={"Avg Sentiment Score": "Avg Score (1-5)", "Model": "EV Model"},
        text="Avg Sentiment Score"
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',
        textfont_color='black',
        marker_line_color='black',
        marker_line_width=1.2
    )

    fig.update_layout(
        font=dict(size=16, color="black"),
        title_font=dict(size=22, color="black"),
        xaxis=dict(
            title=dict(font=dict(color='black', size=18)),
            tickfont=dict(color='black', size=14)
        ),
        yaxis=dict(
            title=dict(font=dict(color='black', size=18)),
            tickfont=dict(color='black', size=14),
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            range=[1, 5]
        ),
        legend=dict(
            font=dict(color='black', size=14)
        ),
        xaxis_tickangle=-30,
        margin=dict(t=70, b=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True
    )

    return fig