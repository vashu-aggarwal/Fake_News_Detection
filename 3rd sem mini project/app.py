import streamlit as st
import pandas as pd
import string
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

vector_form = pickle.load(open('C:/Users/DELL/Downloads/vector (1).pkl', 'rb'))
LR_model = pickle.load(open('C:/Users/DELL/Downloads/LR.pkl', 'rb'))
GB_model = pickle.load(open('C:/Users/DELL/Downloads/GB.pkl', 'rb'))
DT_model = pickle.load(open('C:/Users/DELL/Downloads/DT.pkl', 'rb'))
RF_model = pickle.load(open('C:/Users/DELL/Downloads/RF.pkl', 'rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def fake_news(news, selected_model):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    vector_form1 = vector_form.transform(new_x_test)
    
    if selected_model == "Logistic Regression":
        prediction = LR_model.predict(vector_form1)
    elif selected_model == "Gradient Boosting":
        prediction = GB_model.predict(vector_form1)
    elif selected_model == "Decision Tree":
        prediction = DT_model.predict(vector_form1)
    elif selected_model == "Random Forest":
        prediction = RF_model.predict(vector_form1)
    else:
        prediction = None
    
    return prediction

def home():
    st.title('Btech 3 Semester Mini project 2023')
    st.subheader('Fake News Detection')
    
    st.markdown("<h3 style='color: #0066cc;'>Welcome to the Home Page!</h3>", unsafe_allow_html=True)
    
    st.write("This is a simple app to predict if a given news article is fake or not.")
    st.write("To get started, navigate to the 'Predict' page using the navigation bar on the left.")
    st.write("submited by: Vashu Aggarwal")

def predict_page():
    st.title('Fake News Detection - Predict Page')
    st.markdown(
    f"<style>body {{background-image: linear-gradient(to right, #007BFF, #ADD8E6);}}</style>", unsafe_allow_html=True
     )
    st.markdown("<h3 style='color: #0066cc;'>Enter the news content below:</h3>", unsafe_allow_html=True)
    sentence = st.text_area("Type or paste the news here", "", height=200)
    
    selected_model = st.selectbox("Select Model", ["Logistic Regression", "Gradient Boosting", "Decision Tree", "Random Forest"])

    predict_btt = st.button("Predict", key="predict_button")

    if predict_btt:
        prediction_class = fake_news(sentence, selected_model)
        
        # Display the result with enhanced styling
        st.markdown(
            f"<h3 style='color: {'green' if prediction_class == [0] else 'orange'};'>Prediction:</h3>",
            unsafe_allow_html=True
        )
        if prediction_class == [0]:
            st.success('This is not a fake news.')
        elif prediction_class == [1]:
            st.warning('This is a fake news.')

        # Add a separator
        st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

        # Display the raw prediction values
        st.markdown(
            "<h3 style='color: #0066cc;'>Raw Prediction Values:</h3>",
            unsafe_allow_html=True
        )
        st.write(prediction_class)

        # Add a separator
        st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

        # Display the processed news content
        st.markdown(
            "<h3 style='color: #0066cc;'>Processed News Content:</h3>",
            unsafe_allow_html=True
        )
        st.write(wordopt(sentence))

# Create a simple navigation bar
pages = {"Home": home, "Predict": predict_page}
navigation = st.sidebar.radio("Navigation", list(pages.keys()))

# Display the selected page
pages[navigation]()