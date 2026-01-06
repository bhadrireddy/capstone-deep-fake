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
            background-attachment: fixed;
        }

        header, footer, #MainMenu { visibility: hidden; }

        .hero-title {
            font-size: 8rem !important;
            font-weight: 900 !important;
            color: #1e3a8a !important;
        }

        .hero-subtitle {
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            color: #334155 !important;
        }

        .stButton button {
            background: linear-gradient(135deg, #e05252 0%, #c53030 100%) !important;
            color: white !important;
            padding: 3rem 2rem !important;
            border-radius: 2rem !important;
            font-size: 3rem !important;
            font-weight: 900 !important;
            width: 100%;
        }

        [data-testid="stFileUploader"] {
            padding: 5rem 3rem !important;
            background-color: #ffffff !important;
            border-radius: 2.5rem !important;
            border: 3px dashed #64748b !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #ffffff !important;
            box-shadow: none !important;
        }

        [data-testid="stFileUploader"] button {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        .streamlit-expanderHeader {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            background-color: rgba(255,255,255,0.9) !important;
        }

        [data-testid="stExpander"] {
            background-color: #ffffff !important;
        }

        [data-testid="stExpander"] > div {
            background-color: #ffffff !important;
        }

        .stSelectbox > div > div {
            background-color: #ffffff !important;
        }

        .stSelectbox > div > div > div {
            background-color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Navigation Helpers ---
def go_to_main():
    st.session_state.page = 'main'

def go_to_hero():
    st.session_state.page = 'hero'
    st.session_state.result_data = None

# --- HERO PAGE ---
if st.session_state.page == 'hero':
    _, mid, _ = st.columns([1, 10, 1])
    with mid:
        st.markdown("""
            <div style="text-align:center; padding-top:15vh;">
                <h1 class="hero-title">Deepfake<br>Detector</h1>
                <h2 class="hero-subtitle">Detect AI-generated images and videos</h2>
                <p style="font-size:2.2rem; max-width:70rem; margin:auto;">
                    Upload an image or video to check if it has been manipulated using deepfake techniques.
                </p>
            </div>
        """, unsafe_allow_html=True)

        _, c, _ = st.columns([1, 4, 1])
        with c:
            st.button("START DETECTING ➜", on_click=go_to_main, use_container_width=True)

# --- MAIN PAGE ---
elif st.session_state.page in ["main", "results"]:

    st.markdown("""
        <div style="text-align:center; margin:2rem 0;">
            <h1 style="font-size:5rem; font-weight:900;">Deepfake Detector</h1>
        </div>
    """, unsafe_allow_html=True)

    _, content, _ = st.columns([1, 10, 1])

    with content:
        if st.session_state.result_data is None:

            file_type = st.radio(
                "Select file type",
                ("Image", "Video"),
                horizontal=True,
                label_visibility="collapsed"
            )

            uploaded_file = st.file_uploader(
                f"Drop your {file_type.lower()} here...",
                type=["jpg", "jpeg", "png", "mp4"]
            )

            # ✅ FIX: always define frames
            frames = 0

            with st.expander("Show Advanced Settings"):
                model = st.selectbox(
                    "Select Model",
                    ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
                )
                dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
                threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)

                if file_type == "Video":
                    frames = st.slider("Select Frames", 0, 100, 50)

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
                    "pred": pred,
                    "file_type": file_type
                }
                st.rerun()

        else:
            data = st.session_state.result_data
            st.write(data)

            if st.button("ANALYZE ANOTHER FILE"):
                st.session_state.result_data = None
                st.rerun()
