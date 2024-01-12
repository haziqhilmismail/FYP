import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard for Mat Kilau",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Function to get data from URL and cache it
@st.cache_data
def get_data(url) -> pd.DataFrame:
    return pd.read_csv(url)

# URLs to CSV files
tweets_polarity_url = "https://raw.githubusercontent.com/haziqhilmismail/FYP/main/tweets_polarity.csv"
aspect_with_polarity_url = "https://raw.githubusercontent.com/haziqhilmismail/FYP/main/aspects_with_polarity.csv"

# Read data from URLs
tweets_polarity_df = get_data(tweets_polarity_url)
aspect_with_polarity_df = get_data(aspect_with_polarity_url)

# Filter out "mat" from aspects
aspect_with_polarity_df = aspect_with_polarity_df[aspect_with_polarity_df['Aspect'] != 'mat']

# Dashboard title
st.title("📊 Data Analysis Dashboard")

st.markdown("""This chart shows the frequency of the selected aspects in the dataset. The top 10 aspects are shown as default.""")

top_aspects_list = aspect_with_polarity_df['Aspect'].value_counts().head(10).index.tolist()
selected_aspects = st.multiselect('Select Aspects for Analysis:', aspect_with_polarity_df['Aspect'].unique(), default=top_aspects_list)

col1, col2 = st.columns(2)

with col1:
    # Filter data based on selected aspects
    filtered_aspect_data = aspect_with_polarity_df[aspect_with_polarity_df['Aspect'].isin(selected_aspects)]

    # Count occurrences of each selected aspect
    selected_aspect_counts = filtered_aspect_data['Aspect'].value_counts()

    st.markdown("""### Aspect Frequency Chart""")

    # Horizontal bar chart for the selected aspects
    fig = px.bar(selected_aspect_counts, orientation='h', labels={'value': 'Count', 'index': 'Aspect'})
    st.plotly_chart(fig)
with col2:
    # Prepare data for aspect and polarity distribution chart
    aspect_polarity_counts = aspect_with_polarity_df.groupby(['Aspect', 'Polarity']).size().reset_index(name='Counts')

    # Filter based on selected aspects
    aspect_polarity_counts = aspect_polarity_counts[aspect_polarity_counts['Aspect'].isin(selected_aspects)]

    # Custom color scale
    color_scale = {"Positive": "#90EE90", "Neutral": "#808080", "Negative": "#FF6347"}

    st.markdown("### Aspect Distribution Chart")

    # Create a vertical bar chart for aspect and polarity distribution
    fig2 = px.bar(aspect_polarity_counts, x='Aspect', y='Counts', color='Polarity', barmode='group', color_discrete_map=color_scale)
    st.plotly_chart(fig2)

col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("""### Top 50 Aspect Frequency Table""")
    top_50_aspects_list = aspect_with_polarity_df['Aspect'].value_counts().head(50).reset_index()
    top_50_aspects_list.index = top_50_aspects_list.index + 1
    st.dataframe(top_50_aspects_list, width=300, height=300, hide_index = False)

st.divider()

#justify center for the pie chart
# Count the occurrences of each sentiment
sentiment_counts = tweets_polarity_df['VADER_Sentiment'].value_counts()

st.markdown("""### VADER Sentiment Distribution Chart""")
st.write("""VADER Sentiment: A rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.
         This tool is used to analyze the sentiment of the tweets as a whole. The sentiment is categorized into 3 categories: Positive, Negative and Neutral.
            """)

# Custom color scheme
color_scheme = {"Positive": "#90EE90", "Neutral": "#808080", "Negative": "#FF6347"}
col1, col2, col3 = st.columns([1,2,1])
with col2:
# Create a pie chart using Plotly Express
    fig3 = px.pie(
        sentiment_counts, 
        values=sentiment_counts.values, 
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map=color_scheme
    )

    # Show the pie chart
    st.plotly_chart(fig3)

st.markdown("### VADER Detailed Data View")
st.dataframe(tweets_polarity_df[["Sentence", "VADER_Sentiment"]], use_container_width=True)