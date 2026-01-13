# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained model from Hugging Face (cached)
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
image_url = st.text_input("Enter online image URL", placeholder="https://example.com/image.jpg")

image = None
source = None

# Priority: uploaded file > camera > URL
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        source = "uploaded file"
    except Exception as e:
        st.error(f"Cannot open uploaded file: {e}")

elif camera_image is not None:
    try:
        image = Image.open(camera_image)
        source = "camera capture"
    except Exception as e:
        st.error(f"Cannot process camera image: {e}")

elif image_url.strip():
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(image_url.strip(), headers=headers, timeout=12)
        response.raise_for_status()  # raise exception for bad status codes (4xx, 5xx)

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            st.warning("The URL did not return an image (content-type is not image/*)")
        else:
            image = Image.open(BytesIO(response.content))
            source = "URL"
    except requests.exceptions.Timeout:
        st.error("Request timed out – the server took too long to respond.")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err} (status code {response.status_code if 'response' in locals() else 'unknown'})")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Failed to download image: {req_err}")
    except Exception as e:
        st.error(f"Cannot open image from URL: {e}")

if image is not None:
    st.image(image, caption=f"Input Image ({source})", use_column_width=True)
    
    if st.button("Generate Description"):
        with st.spinner("Generating description..."):
            try:
                result = model(image)[0]['generated_text']
                st.success("Description generated!")
                st.markdown("**Description:** " + result)
            except Exception as e:
                st.error(f"Model failed to process image: {e}")
else:
    st.info("Please provide an image using one of the options above.")
