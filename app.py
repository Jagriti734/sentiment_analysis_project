import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("🎬 Sentiment Analysis System")

review = st.text_area("Enter a movie review")

if st.button("Predict"):

    if review.strip():

        review_vector = tfidf.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")