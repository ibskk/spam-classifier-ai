import streamlit as st
from model import train_model

# Train the model when the app starts
model, vectorizer, accuracy = train_model()

st.title("Spam Detection AI")
st.write(f"Model accuracy: {accuracy:.2f}")

user_input = st.text_area("Enter a message")

if st.button("Predict"):
    vector = vectorizer.transform([user_input])
    prediction = model.predict(vector)
    st.write("Prediction:", prediction[0])
