import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

@st.cache_resource
def load_model():
    assert os.path.exists("mnist_model.keras"), "Model file not found!"
    return tf.keras.models.load_model("mnist_model.keras")

model = load_model()

st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload a **28x28 grayscale image** of a digit (0–9) to classify.")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    st.image(img, caption="Input Image", use_column_width=True)

    input_data = img.reshape(1, 28, 28, 1).astype("float32") / 255.0
    prediction = model.predict(input_data)
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    st.success(f"Predicted Digit: {predicted_label}")
