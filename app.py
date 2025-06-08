# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# # Page config
# st.set_page_config(page_title="InspectorsAlly – Banana QA", layout="centered")

# # Load the model once
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("model/")


# model = load_model()
# class_names = ["Normal", "Defective"]

# # Header
# st.markdown("""
# <h1 style='text-align: center;'>🍌 InspectorsAlly – Banana Quality Inspection</h1>
# <p style='text-align: center;'>AI-powered tool to detect defects in bananas</p>
# <hr>
# """, unsafe_allow_html=True)

# # Dataset preview
# st.subheader("📂 Sample Dataset Preview")
# st.image("docs/goodvsbad.png", caption="Normal vs Defective Bananas", use_container_width=True)

# # Input method
# input_method = st.radio("Choose Input Method", ["📤 Upload Image", "📷 Use Camera"], horizontal=True)
# image = None

# # Confidence threshold slider
# confidence_threshold = st.slider("Confidence Threshold", 0, 100, 50, 1)

# # Input
# if input_method == "📤 Upload Image":
#     uploaded_file = st.file_uploader("Upload banana image", type=["jpg", "jpeg", "png"], key="file_uploader")
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")

# elif input_method == "📷 Use Camera":
#     camera_input = st.camera_input("Capture banana using webcam", key="camera_input")
#     if camera_input:
#         image = Image.open(camera_input).convert("RGB")

# # Prediction
# if image:
#     st.image(image, caption="📷 Input Image", use_container_width=True)

#     image = image.resize((224, 224))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = float(np.max(prediction)) * 100
#     label_color = "green" if predicted_class == "Normal" else "red"
#     emoji = "✅" if predicted_class == "Normal" else "❌"

#     st.markdown(f"""
#     <div style='background-color:#0e1117; padding:20px; border-radius:10px; box-shadow:0 0 10px rgba(255,255,255,0.05);'>
#         <h3 style='color:{label_color}; text-align:center;'>{emoji} Prediction: {predicted_class}</h3>
#         <p style='text-align:center;'>Confidence: <b>{confidence:.2f}%</b></p>
#         <progress value="{confidence}" max="100" style="width: 100%; height: 20px; border-radius: 8px;"></progress>
#     </div>
#     """, unsafe_allow_html=True)

#     if confidence < confidence_threshold:
#         st.warning(f"⚠️ Confidence below threshold ({confidence_threshold}%) – consider retesting")

#     if st.button("🔄 Try Another Image"):
#         for key in ["file_uploader", "camera_input"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.rerun()

# # Expandable references
# with st.expander("📘 Model & Deployment References"):
#     st.markdown("""
#     - [Teachable Machine](https://teachablemachine.withgoogle.com/)
#     - [Streamlit Docs](https://docs.streamlit.io/)
#     - [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
#     - [30 Days of Streamlit](https://30days.streamlit.app/)
#     """)

# st.markdown("---")
# st.caption("Built with ❤️ by Sujal • Powered by TensorFlow + Streamlit")



import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="InspectorsAlly – Banana QA", layout="centered")

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

model = load_model()

# Load class labels from Teachable Machine
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Header
st.markdown("""
<h1 style='text-align: center;'>🍌 InspectorsAlly – Banana Quality Inspection</h1>
<p style='text-align: center;'>AI-powered tool to detect defects in bananas</p>
<hr>
""", unsafe_allow_html=True)

# Dataset preview
st.subheader("📂 Sample Dataset Preview")
st.image("docs/goodvsbad.png", caption="Normal vs Defective Bananas", use_container_width=True)

# Input method
input_method = st.radio("Choose Input Method", ["📤 Upload Image", "📷 Use Camera"], horizontal=True)
image = None

# Confidence threshold slider
confidence_threshold = st.slider("Confidence Threshold", 0, 100, 50, 1)

# Input
if input_method == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload banana image", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif input_method == "📷 Use Camera":
    camera_input = st.camera_input("Capture banana using webcam", key="camera_input")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")

# Prediction
if image:
    st.image(image, caption="📷 Input Image", use_container_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    label_color = "green" if predicted_class == "Normal" else "red"
    emoji = "✅" if predicted_class == "Normal" else "❌"

    st.markdown(f"""
    <div style='background-color:#0e1117; padding:20px; border-radius:10px; box-shadow:0 0 10px rgba(255,255,255,0.05);'>
        <h3 style='color:{label_color}; text-align:center;'>{emoji} Prediction: {predicted_class}</h3>
        <p style='text-align:center;'>Confidence: <b>{confidence:.2f}%</b></p>
        <progress value="{confidence}" max="100" style="width: 100%; height: 20px; border-radius: 8px;"></progress>
    </div>
    """, unsafe_allow_html=True)

    if confidence < confidence_threshold:
        st.warning(f"⚠️ Confidence below threshold ({confidence_threshold}%) – consider retesting")

    if st.button("🔄 Try Another Image"):
        for key in ["file_uploader", "camera_input"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Expandable references
with st.expander("📘 Model & Deployment References"):
    st.markdown("""
    - [Teachable Machine](https://teachablemachine.withgoogle.com/)
    - [Streamlit Docs](https://docs.streamlit.io/)
    - [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
    - [30 Days of Streamlit](https://30days.streamlit.app/)
    """)

st.markdown("---")
st.caption("Built with ❤️ by Sujal • Powered by TensorFlow + Streamlit")



