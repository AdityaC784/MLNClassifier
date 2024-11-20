import streamlit as st
import joblib

# Load models
with open('Multinomial.pkl', 'rb') as f:
    english_model = joblib.load(f)

with open('hindi_Rf.pkl', 'rb') as f:
    hindi_model = joblib.load(f)

# Load TF-IDF vectorizers
with open('tokenization.pkl', 'rb') as f:
    english_tfidf = joblib.load(f)

with open('hindiTokenization.pkl', 'rb') as f:
    hindi_tfidf = joblib.load(f)

categories = {
    0: "Entertainment",
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Science",
    5: "National",
    6: "Miscellaneous"
}

st.title("News Classification")
st.header("Choose the type of news you want to classify")

# Language selection
language = st.selectbox("Select Language", ("English", "Hindi"))

# Text area for multiple news entries, separated by new lines
news_text = st.text_area("Enter the news texts here (one per line)")

if st.button("Classify News"):
    # Split the input text into individual news items
    news_list = news_text.splitlines()

    predictions = []
    
    for news in news_list:
        if language == "English":
            transformed_text = english_tfidf.transform([news])
            prediction = english_model.predict(transformed_text)[0]
        else:
            transformed_text = hindi_tfidf.transform([news])
            prediction = hindi_model.predict(transformed_text)[0]
        
        category_label = categories.get(prediction, "Unknown Category")
        predictions.append(category_label)
    
    # Display results for each news item
    for i, news in enumerate(news_list):
        st.write(f"News: {news}")
        st.write(f"Classified as: {predictions[i]}")
