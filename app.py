import streamlit as st

st.title("Spam Detection AI")

user_input = st.text_area("Enter a message")

if st.button("Predict"):
    vector = vectorizer.transform([user_input])
    prediction = model.predict(vector)
    st.write("Prediction:", prediction[0])
