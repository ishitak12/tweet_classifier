import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict sentiment
def predict_sentiment(tweet):
    tweet_vect = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vect)
    return prediction[0]

def predict_bulk(tweets):
    tweet_vect = vectorizer.transform(tweets)
    predictions = model.predict(tweet_vect)
    return predictions

# Streamlit app UI
st.title("Tweet Sentiment Classifier")
#st.markdown("This app classifies tweets as **Positive** or **Negative** using a trained machine learning model.")

# Tabs for single and bulk tweet analysis
tab1, tab2 = st.tabs(["🔹 Single Tweet", "🔸 Multiple Tweets"])

# --- Single Tweet Tab ---
with tab1:
    tweet = st.text_area("Enter a single tweet:")
    if st.button("Classify"):
        if tweet.strip():
            sentiment = "Positive" if predict_sentiment(tweet) == 1 else "Negative"
            st.success(f"Sentiment: **{sentiment}**")
        else:
            st.warning("Please enter a tweet to classify.")

# --- Bulk Tweets Tab ---
with tab2:
    st.markdown("### Option 1: Paste multiple tweets (one per line)")
    bulk_input = st.text_area("Enter multiple tweets here, one per line:")
    
    st.markdown("### Option 2: Upload a CSV file with a column named `tweet`")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if st.button("Analyze Multiple Tweets"):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "tweet" not in df.columns:
                st.error("CSV must contain a 'tweet' column.")
            else:
                predictions = predict_bulk(df["tweet"])
                df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in predictions]
                st.dataframe(df[["tweet", "Sentiment"]])
                st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")
        elif bulk_input.strip():
            tweets_list = [line.strip() for line in bulk_input.strip().split("\n") if line.strip()]
            predictions = predict_bulk(tweets_list)
            results = pd.DataFrame({
                "Tweet": tweets_list,
                "Sentiment": ["Positive" if p == 1 else "Negative" for p in predictions]
            })
            st.dataframe(results)
            st.download_button("Download Results", results.to_csv(index=False), "predictions.csv", "text/csv")
        else:
            st.warning("Please upload a CSV or enter multiple tweets.")
