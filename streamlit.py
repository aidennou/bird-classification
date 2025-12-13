import streamlit as st
from fastai.vision.all import load_learner, PILImage

st.title("Bird Species Classifier")
st.text("Built by Aiden Ou")

learn = load_learner("bird_classification.pkl")

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = learn.predict(img)
    confidence = outputs[pred_idx].item()
    return f"{pred_class} ({confidence:.2f} confidence)"

uploaded_file = st.file_uploader("Upload a bird image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prediction = predict(uploaded_file)
    st.write("Prediction:", prediction)
