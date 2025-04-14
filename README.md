<h1># Tweet Sentiment Classifier 🐦💬</h1>

<h2 style="font-size: 36px;">Welcome to the Tweet Sentiment Classifier! 🐦💬</h2>

## Overview 🌟

Welcome to the **Tweet Sentiment Classifier**! This is a Python-based machine learning application that analyzes tweets and classifies them into **Positive** or **Negative** sentiment based on the text.

### 🚀 Features:
- Predict sentiment of tweets as **Positive** or **Negative**.
- Built using **Scikit-learn** and **Pandas**.

---

## Setup Instructions ⚙️

### Prerequisites:
Make sure you have **Python 3.x** installed. 🐍

### Steps to Set Up:
1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/ishitak12/tweet_classifier.git
2. Install dependencies:
Make sure you have Python 3.x installed, then install the required dependencies:

pip install -r requirements.txt

Add Dataset to the data folder: Note: The dataset is too large to upload to GitHub, so it is ignored via .gitignore. You can manually add the dataset to the data/ folder.

<h3>Dataset link: Download the dataset from [Sentiment140](http://www.sentiment140.com/)</h3>.

After downloading, place the CSV file inside the data/ folder.

3.Train the Model (Optional) 🔬
If you'd like to train the sentiment analysis model yourself, run the train_model.py script:

This will train the model and save the sentiment_model.pkl and vectorizer.pkl files that are used to classify tweets.
4.Make Predictions 🧠
Once the model is trained, you can use the app.py script to classify new tweets.
