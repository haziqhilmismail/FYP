import streamlit as st
import pandas as pd
from wordcloud import WordCloud
from pyabsa import AspectTermExtraction as ATEPC
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="ABSA for Mat Kilau",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the state dict
state_dict_path = 'state_dict_model_FACT_LCF_ATEPC/fast_lcf_atepc_custom_dataset_cdw_apcacc_61.54_apcf1_61.44_atef1_69.33/'

# Load the model using the checkpoint manager
aspect_classifier = ATEPC.AspectExtractor(state_dict_path, auto_device=True)

sid = SentimentIntensityAnalyzer()

# find a suitable checkpoint and use the name:
# aspect_classifier = ATEPC.AspectExtractor(
#     checkpoint="english"
# )  
   
# CSS to inject contained in a string
# css = """
#             <style>
#             thead tr th:first-child {display:none}
#             tbody th {display:none}
#             #MainMenu {visibility:hidden;}
#             footer {visibility:hidden;}
#             </style>
#             """

# Inject CSS with Markdown
# st.markdown(css, unsafe_allow_html=True)

def colorize(val):
    color = "#90EE90" if val == 'Negative' else "#FF6347" if val == 'Positive' else "#808080"
    return f'background-color: {color}'

# App Header
st.title('📊 ABSA for Mat Kilau Movie Reviews')
st.subheader('Discover the sentiment of your review')
st.markdown('''
    This application was made using **PyABSA**. You can analyze the sentiment of different aspects of your text.
    Just paste your text in the text box and click on the 'Analyze' button. For a better result,
    please use a review with aspect that is related to the movie. You can use wordcloud below to see the most common aspects.

    Example: **The actor is great but sadly the camera shaking is too much**
''')

# Sidebar for user instruction
with st.expander("User Instruction:"):
    st.markdown('''
    1. Input your review text in the text box below.
    2. Click on the 'Analyze' button.
    3. Wait for a few moments to see the result.
    4. The result will show the extracted aspects, their sentiment and confidence.
    ''')

# Get user input
examples = st.text_input('Enter your review here...')

# Button to perform sentiment analysis
# Button to perform sentiment analysis
if st.button('Analyze'):
    if examples:
        with st.spinner("Analyzing..."):

            inference_source = [examples]
            atepc_result = aspect_classifier.predict(
                                    text=inference_source,
                                    print_result=False,
                                    save_result=False,
                                    ignore_error=True,  # Predict the sentiment of extracted aspect terms
                                )

            # Convert result to data frame for better visual in Streamlit
            processed_result = []
            for item in atepc_result:
                for aspect, sentiment, confidence in zip(item['aspect'], item['sentiment'], item['confidence']):
                    processed_result.append({'Aspect': aspect, 'Sentiment': sentiment, 'Confidence': confidence})

            vader_scores = sid.polarity_scores(examples)
            vader_sentiment = 'Positive' if vader_scores['compound'] >= 0.05 else 'Negative' if vader_scores['compound'] <= -0.05 else 'Neutral'
            
            if vader_scores:
                # Data frame for the bar chart
                vader_df_chart = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Score': [vader_scores['neg'], vader_scores['neu'], vader_scores['pos']],
                    'Color': ['#FF6347', '#808080', '#90EE90']  # Red, Grey, Green
                })

                fig = px.bar(
                    vader_df_chart, 
                    x='Sentiment', 
                    y='Score', 
                    color='Color', 
                    color_discrete_map="identity"
                )

                fig.update_layout(showlegend=False)

                # Use Streamlit columns to align text and chart
                col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed

                with col1:
                    st.markdown("### Sentiment Distribution Result")
                    if processed_result:
                        result_df = pd.DataFrame(processed_result)
                        st.dataframe(result_df.style.applymap(colorize, subset=['Sentiment']))
                        annotated_text = ""
                        for word in examples.split():
                            sentiment_found = False
                            for result in processed_result:
                                if word.lower() == result['Aspect'].lower():
                                    color = "#90EE90" if result['Sentiment'] == "Positive" else "#FF6347" if result['Sentiment'] == "Negative" else "#808080"
                                    annotated_text += f"<span style='color: {color}'>{word}</span> "
                                    sentiment_found = True
                                    break
                            if not sentiment_found:
                                annotated_text += word + " "
                        st.markdown(annotated_text, unsafe_allow_html=True)
                    else:
                        st.write("No aspects detected in the input.")
                with col2:
                    st.markdown("### Vader Sentence Sentiment Result")
                    sentiment_color = "#90EE90" if vader_sentiment == "Positive" else "#FF6347" if vader_sentiment == "Negative" else "#808080"
                    st.markdown(f"#### Overall Sentiment (VADER): <span style='color: {sentiment_color};'>{vader_sentiment}</span>", unsafe_allow_html=True)
                    st.plotly_chart(fig)
                    
    else:
        st.write("Please enter a review.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Positive Aspects")
    st.image("Positive.png")

with col2:
    st.markdown("### Neutral Aspects")
    st.image("Neutral.png")

with col3:
    st.markdown("### Negative Aspects")
    st.image("Negative.png")

# Function to get data for word cloud
# @st.cache_data
# def get_wordcloud_data(url):
#     df = pd.read_csv(url)
#     return df

# aspect_url = "https://raw.githubusercontent.com/haziqhilmismail/FYP/main/aspects_with_polarity.csv"
# aspect_df = get_wordcloud_data(aspect_url)

# # Function to generate word cloud data for a specific sentiment
# def create_wordcloud(df, sentiment):
#     filtered_df = df[df['Polarity'] == sentiment]
#     text = ' '.join(filtered_df['Aspect'])
#     return text

# # Generate word cloud text for each sentiment
# positive_text = create_wordcloud(aspect_df, 'Positive')
# neutral_text = create_wordcloud(aspect_df, 'Neutral')
# negative_text = create_wordcloud(aspect_df, 'Negative')

# # Function to display word cloud
# def display_wordcloud(text, color):
#     wordcloud = WordCloud(background_color='white', colormap=color, contour_color=color, contour_width=3).generate(text)
#     fig, ax = plt.subplots(figsize=(20, 10))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot(fig)



