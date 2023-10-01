# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:16:49 2023
@author: DELL
"""
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from streamlit import set_page_config

# Load the saved Random Forest model
pickle_in = open(r"C:\Users\DELL\Desktop\P-272\rf_model.pkl", "rb")
rf_model = pickle.load(pickle_in)

# Load the TF-IDF vectorizer if used during training
tfidf_vectorizer = pickle.load(open(r"C:\Users\DELL\Desktop\P-272\tfidf_vectorizer.pkl", "rb"))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    processed_text = ' '.join(words)
    return processed_text

def predict_note_authentication(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Transform the preprocessed text using TF-IDF
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    
    # Make predictions using the model
    prediction = rf_model.predict(text_tfidf)
    
    # Convert numeric prediction to labels
    label = "Fake" if prediction == 0 else "True"
    return label

def main():
    # Set page configuration
    set_page_config(
        page_title="News Authentication App",
        page_icon="📰",
        layout="wide"
    )

    # Use CSS to set background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpapers.com/images/hd/news-studio-background-76xgylidicak0vju.jpg');
            background-size: cover;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("News Authentication")

    # Display the additional image using st.image
    # img = Image.open("news-studio.jpg")
    # st.image(img)

    text = st.text_input("Text", "Type Here")
    
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(text)
    
    st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()
