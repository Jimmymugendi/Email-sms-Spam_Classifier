import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#load the saved vectorizer and naivebayes model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

#transform_text function for preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()

def transform_message(message):
  message = message.lower() #converting to lowercase
  message = nltk.word_tokenize(message) #tokenize

  #remove special characters and retaining alphanumeric words
  message = [word for word in message if word.isalnum()]

  #remove the punctuation and stop wordsstreamlit
  message = [word for word in message if word in stopwords.words('english') and word not in string.punctuation]

  #apply stemming
  message = [ps.stem(word) for word in message]

  return " ".join(message)

#saving the streamlit code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
  #preprocess
  transformed_sms = transform_message(input_sms)
  #vectorize
  vector_input = tfidf.transform([transformed_sms])
  #predict
  result = model.predict(vector_input)[0]
  #display
  if result == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")


