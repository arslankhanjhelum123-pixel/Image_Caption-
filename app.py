# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# ────────────────────────────────────────────────
# Page config - makes it look more professional
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Describer – BLIP Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Load model once (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

model = load_model()

# ────────────────────────────────────────────────
# Sidebar – all controls here
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("🖼️ Upload or Capture Image")
    st.markdown("Choose **one** of the options below:")

    # File upload
    uploaded_file = st.file_uploader(
        "1. Upload from device",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported: JPG, PNG, WEBP"
    )

    # Camera
    st.markdown("**— or —**")
    camera_image = st.camera_input(
        "2. Take photo with camera",
        help="Use your webcam or phone camera"
    )

    # URL
    st.markdown("**— or —**")
    image_url = st.text_input(
        "3. Paste direct image URL",
        placeholder="https://example.com/image.jpg",
        help="Must be a direct link to .jpg / .png / etc."
    )

    st.markdown("---")
    st.caption("Tip: Only one input is used (upload > camera > URL)")

# ────────────────────────────────────────────────
# Main area
# ────────────────────────────────────────────────
st.title("🖼️ Image Description Generator")
st.markdown("Get automatic captions using **BLIP** – a free, open-source model from Salesforce.")

# Process image
image = None
source = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    source = "Uploaded file"
elif camera_image is not None:
    image = Image.open(camera_image)
    source = "Camera capture"
elif image_url.strip():
    with st.spinner("Fetching image from URL..."):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
            }
            resp = requests.get(image_url.strip(), headers=headers, timeout=12)
            resp.raise_for_status()

            if "image" not in resp.headers.get("Content-Type", "").lower():
                st.sidebar.error("URL does not point to an image file.")
            else:
                image = Image.open(BytesIO(resp.content))
                source = "Web URL"
        except requests.Timeout:
            st.sidebar.error("⏱️ Timeout – server too slow")
        except requests.HTTPError as e:
            st.sidebar.error(f"🌐 HTTP {e.response.status_code} error")
        except Exception as e:
            st.sidebar.error(f"⚠️ Could not load URL: {e}")

# ────────────────────────────────────────────────
# Tabs for clean separation
# ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Input Image", "📝 Description"])

with tab1:
    if image:
        st.image(image, caption=f"Source: {source}", use_column_width=True)
    else:
        st.info("Please provide an image using the sidebar controls ↑")

with tab2:
    if image:
        if st.button("✨ Generate Description", type="primary", use_container_width=True):
            with st.spinner("Thinking... (BLIP model is running)"):
                try:
                    result = model(image)[0]["generated_text"]
                    st.success("Description ready!")
                    st.markdown(f"**Caption:** {result.capitalize()}")
                    
                    # Optional: copy button
                    st.code(result, language=None)
                except Exception as e:
                    st.error(f"Model error: {e}")
    else:
        st.info("Upload/capture/paste an image first to enable description.")

# Footer
st.markdown("---")
st.caption("Powered by Hugging Face • Salesforce/blip-image-captioning-base • Built with Streamlit")
