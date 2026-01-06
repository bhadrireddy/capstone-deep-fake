import streamlit as st
from PIL import Image
from api import process_image, process_video
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Session State Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'hero'
if 'result_data' not in st.session_state:
    st.session_state.result_data = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = "Image"

# Ensure uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# --- CSS & Styling Injection ---
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6, span, div, label, p {
    color: #0f172a !important;
}

.stApp {
    background: linear-gradient(180deg, #FFFBF0 0%, #FFF5E6 100%);
}

header, footer, #MainMenu {
    visibility: hidden;
}

/* ---------- HERO ---------- */
.hero-title {
    font-size: 8rem !important;
    font-weight: 900 !important;
    color: #1e3a8a !important;
}

.hero-subtitle {
    font-size: 3.5rem !important;
    font-weight: 700 !important;
}

/* ---------- BUTTONS ---------- */
.stButton button {
    background: linear-gradient(135deg, #e05252 0%, #c53030 100%) !important;
    color: white !important;
    padding: 3rem 2rem !important;
    border-radius: 2rem !important;
    font-size: 3rem !important;
    font-weight: 900 !important;
}

/* ---------- RADIO BUTTONS (IMAGE / VIDEO HUGE) ---------- */
.stRadio label {
    font-size: 4rem !important;
    padding: 3rem 6rem !important;
    background: rgba(255,255,255,0.9);
    border-radius: 4rem;
    border: 5px solid #94a3b8;
}

.stRadio input {
    transform: scale(3);
}

/* ---------- FILE UPLOADER ---------- */
[data-testid="stFileUploader"] {
    min-height: 20rem !important;
    padding: 6rem 4rem !important;
    border: 6px dashed #64748b;
    border-radius: 2rem;
    background: rgba(255,255,255,0.7);
}

/* Drop your image/video here */
[data-testid="stFileUploader"] label {
    font-size: 3rem !important;
    font-weight: 800 !important;
}

/* Drag & drop text */
[data-testid="stFileUploader"] div div {
    font-size: 2.5rem !important;
    color: #1e40af !important;
    font-weight: 700;
}

/* ---------- ADVANCED SETTINGS ---------- */
.streamlit-expanderHeader {
    font-size: 6rem !important;
    font-weight: 900 !important;
    background: rgba(255,255,255,0.8);
}

/* Prevent black background */
.streamlit-expanderContent {
    background: rgba(255,255,255,0.95) !important;
    color: #0f172a !important;
}

/* Remove empty white box */
.streamlit-expanderContent > div:empty {
    display: none !important;
}

/* ---------- SELECT MODEL ---------- */
.stSelectbox label {
    font-size: 4rem !important;
    font-weight: 800;
}

.stSelectbox div[data-baseweb="select"] {
    font-size: 3rem !important;
    min-height: 6rem !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #1e3a8a !important;
    font-weight: 700;
}

/* ---------- DATASET (DFDC / FFPP) ---------- */
.stRadio div[role="radiogroup"] label {
    font-size: 3.5rem !important;
    padding: 2.5rem 4rem !important;
}

/* ---------- SLIDERS ---------- */
.stSlider label {
    font-size: 4rem !important;
    font-weight: 800;
}

.stSlider div[data-baseweb="slider"] {
    padding: 3rem 0 !important;
}

button[title="View fullscreen"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# --- Navigation ---
def go_to_main():
    st.session_state.page = 'main'

# --- HERO PAGE ---
if st.session_state.page == 'hero':
    _, mid, _ = st.columns([1, 10, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center; padding-top:15vh;">
            <h1 class="hero-title">Deepfake<br>Detector</h1>
            <h2 class="hero-subtitle">Detect AI-generated images and videos</h2>
            <p style="font-size:2rem;">
                Upload an image or video to check if it has been manipulated.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.button("START DETECTING ➜", on_click=go_to_main, use_container_width=True)

# --- MAIN PAGE ---
else:
    _, content, _ = st.columns([1, 10, 1])
    with content:

        if st.session_state.result_data is None:
            file_type = st.radio(
                "Select File Type",
                ("Image", "Video"),
                horizontal=True,
                label_visibility="collapsed"
            )

            uploaded_file = st.file_uploader(
                f"Drop your {file_type.lower()} here...",
                type=["jpg", "jpeg", "png", "mp4"]
            )

            with st.expander("Show Advanced Settings"):
                model = st.selectbox(
                    "Select Model",
                    (
                        "EfficientNetB4",
                        "EfficientNetB4ST",
                        "EfficientNetAutoAttB4",
                        "EfficientNetAutoAttB4ST"
                    )
                )
                dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
                threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
                frames = st.slider("Select Frames", 0, 100, 50) if file_type == "Video" else 0

            if uploaded_file and st.button("CHECK FOR DEEPFAKE", use_container_width=True):
                with st.spinner("Analyzing media..."):
                    time.sleep(1)

                    if file_type == "Image":
                        result, pred = process_image(
                            image=uploaded_file,
                            model=model,
                            dataset=dataset,
                            threshold=threshold
                        )
                    else:
                        path = f"uploads/{uploaded_file.name}"
                        with open(path, "wb") as f:
                            f.write(uploaded_file.read())

                        result, pred = process_video(
                            path,
                            model=model,
                            dataset=dataset,
                            threshold=threshold,
                            frames=frames
                        )

                st.session_state.result_data = {
                    "result": result,
                    "pred": pred
                }
                st.rerun()
