# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained model from Hugging Face
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

model = load_model()

st.title("Image Description Generator")

st.write("Upload an image, capture from camera, or provide an online image link to generate a description.")

# Option 1: Upload image from device
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Option 2: Capture live from camera
camera_image = st.camera_input("Capture from camera")

# Option 3: Online image link
image_url = st.text_input("Enter online image URL")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)
elif image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)
    if st.button("Generate Description"):
        with st.spinner("Generating description..."):
            result = model(image)[0]['generated_text']
            st.success("Description generated!")
            st.write(result)
else:
    st.info("Please provide an image using one of the options above.")
