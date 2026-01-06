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
st.markdown(
    """
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

        header, footer, #MainMenu {
            visibility: hidden;
        }

        .hero-title {
            font-size: 8rem !important;
            font-weight: 900 !important;
            line-height: 1.1;
            letter-spacing: -0.02em;
            color: #1e3a8a !important;
            text-shadow: 2px 2px 0px rgba(255,255,255,0.5);
            margin-bottom: 1rem;
        }

        .hero-subtitle {
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            color: #334155 !important;
            margin-bottom: 3rem;
        }

        .stButton button {
            background: linear-gradient(135deg, #e05252 0%, #c53030 100%) !important;
            color: white !important;
            border: none;
            padding: 3rem 2rem !important;
            border-radius: 2rem !important;
            font-weight: 900 !important;
            font-size: 3rem !important;
            width: 100%;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        .stButton button p {
            color: white !important;
            font-size: 3rem !important;
        }

        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }

        [data-testid="stFileUploader"] {
            padding: 4rem 2rem !important;
            background-color: rgba(255,255,255,0.6);
            border: 4px dashed #64748b;
            border-radius: 2rem;
        }

        .stRadio > div {
            flex-direction: row;
            gap: 50px;
            justify-content: center;
        }

        .streamlit-expanderHeader {
            font-size: 25rem !important;
            font-weight: 700 !important;
            background-color: rgba(255,255,255,0.5);
            border-radius: 1rem;
            padding: 1.5rem !important;
        }

        button[title="View fullscreen"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
def go_to_main():
    st.session_state.page = 'main'

def go_to_hero():
    st.session_state.page = 'hero'
    st.session_state.result_data = None

# --- HERO PAGE ---
if st.session_state.page == 'hero':
    col1, col2, col3 = st.columns([1, 10, 1])

    with col2:
        st.markdown(
            """
            <div style="text-align:center; padding-top:15vh;">
                <h1 class="hero-title">Deepfake<br>Detector</h1>
                <h2 class="hero-subtitle">Detect AI-generated images and videos</h2>
                <p style="font-size:2rem; max-width:70rem; margin:auto;">
                    Upload an image or video to check if it has been manipulated using deepfake techniques.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        _, mid, _ = st.columns([1, 4, 1])
        with mid:
            st.button("START DETECTING ➜", on_click=go_to_main, use_container_width=True)

# --- MAIN PAGE ---
elif st.session_state.page in ['main', 'results']:
    st.markdown(
        """
        <div style="text-align:center; margin:2rem 0;">
            <h1 style="font-size:5rem; font-weight:900;">Deepfake Detector</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

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
