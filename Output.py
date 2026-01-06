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
if 'dataset' not in st.session_state:
    st.session_state.dataset = "DFDC"

# Ensure uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# --- CSS & Styling Injection ---
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <script>
        // Force radio buttons to be clickable
        window.addEventListener('load', function() {
            setTimeout(function() {
                const radioInputs = document.querySelectorAll('[data-testid="stExpander"] input[type="radio"]');
                radioInputs.forEach(function(radio) {
                    radio.style.pointerEvents = 'auto';
                    radio.style.cursor = 'pointer';
                    radio.style.zIndex = '9999';
                    radio.style.position = 'relative';
                    radio.addEventListener('click', function(e) {
                        e.stopPropagation();
                    });
                });
                
                const radioLabels = document.querySelectorAll('[data-testid="stExpander"] .stRadio label');
                radioLabels.forEach(function(label) {
                    label.style.pointerEvents = 'auto';
                    label.style.cursor = 'pointer';
                    label.style.zIndex = '9999';
                    label.style.position = 'relative';
                });
            }, 100);
        });
    </script>
    
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
            border: 3px dashed #000000 !important;
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

        [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
            background-color: #ffffff !important;
        }

        [data-testid="stExpander"] [data-testid="stVerticalBlock"] > div {
            background-color: #ffffff !important;
        }

        [data-testid="stExpander"] .element-container {
            background-color: #ffffff !important;
        }

        /* Removed universal selector to prevent blocking radio button clicks */
        [data-testid="stExpander"] > div,
        [data-testid="stExpander"] [data-testid="stVerticalBlock"],
        [data-testid="stExpander"] [data-testid="stVerticalBlock"] > div,
        [data-testid="stExpander"] .element-container,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {
            background-color: #ffffff !important;
        }

        /* Ensure no element blocks radio button clicks */
        [data-testid="stExpander"] .stRadio,
        [data-testid="stExpander"] .stRadio * {
            pointer-events: auto !important;
        }

        [data-testid="stExpander"] label {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        [data-testid="stExpander"] .stSelectbox {
            background-color: #ffffff !important;
        }

        [data-testid="stExpander"] .stRadio {
            background-color: #ffffff !important;
            pointer-events: auto !important;
            position: relative !important;
            z-index: 1000 !important;
        }

        /* Remove any overlays that might block clicks */
        [data-testid="stExpander"]::before,
        [data-testid="stExpander"]::after {
            display: none !important;
            pointer-events: none !important;
        }

        /* Ensure expander content doesn't block */
        [data-testid="stExpander"] [data-testid="stExpanderContent"] {
            pointer-events: auto !important;
        }

        [data-testid="stExpander"] [data-testid="stExpanderContent"] * {
            pointer-events: auto !important;
        }

        /* Radio button styling to ensure clickability - override universal selector */
        [data-testid="stExpander"] .stRadio,
        [data-testid="stExpander"] .stRadio *,
        [data-testid="stExpander"] .stRadio input,
        [data-testid="stExpander"] .stRadio label,
        [data-testid="stExpander"] .stRadio [data-baseweb="radio"],
        [data-testid="stExpander"] .stRadio [role="radio"],
        [data-testid="stExpander"] .stRadio [role="radiogroup"] {
            pointer-events: auto !important;
            cursor: pointer !important;
            z-index: 999 !important;
            position: relative !important;
        }

        .stRadio > div {
            display: flex !important;
            flex-direction: column !important;
            gap: 1rem !important;
            pointer-events: auto !important;
            position: relative !important;
            z-index: 10 !important;
        }

        .stRadio label {
            display: flex !important;
            align-items: center !important;
            cursor: pointer !important;
            pointer-events: auto !important;
            padding: 0.5rem !important;
            background-color: #ffffff !important;
            position: relative !important;
            z-index: 10 !important;
        }

        .stRadio input[type="radio"] {
            cursor: pointer !important;
            pointer-events: auto !important;
            width: 20px !important;
            height: 20px !important;
            margin-right: 0.75rem !important;
            accent-color: #dc2626 !important;
            position: relative !important;
            z-index: 11 !important;
            opacity: 1 !important;
        }

        .stRadio input[type="radio"]:checked {
            accent-color: #dc2626 !important;
        }

        /* Ensure radio button container is clickable */
        .stRadio [data-baseweb="radio"] {
            pointer-events: auto !important;
            cursor: pointer !important;
        }

        .stRadio [data-baseweb="radio"] > div {
            pointer-events: auto !important;
            cursor: pointer !important;
        }

        /* Force radio buttons to be interactive */
        [data-testid="stExpander"] .stRadio [role="radiogroup"] {
            pointer-events: auto !important;
        }

        [data-testid="stExpander"] .stRadio [role="radio"] {
            pointer-events: auto !important;
            cursor: pointer !important;
            z-index: 100 !important;
            position: relative !important;
        }

        [data-testid="stExpander"] .stRadio label[data-baseweb="radio"] {
            pointer-events: auto !important;
            cursor: pointer !important;
            z-index: 100 !important;
        }

        [data-testid="stExpander"] .stSlider {
            background-color: #ffffff !important;
        }

        .stSelectbox > div > div {
            background-color: #ffffff !important;
        }

        .stSelectbox > div > div > div {
            background-color: #ffffff !important;
        }

        .stSelectbox [role="listbox"] {
            background-color: #ffffff !important;
        }

        .stSelectbox [role="option"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        .stSelectbox [role="option"]:hover {
            background-color: #f1f5f9 !important;
        }

        div[data-baseweb="select"] {
            background-color: #ffffff !important;
        }

        div[data-baseweb="popover"] {
            background-color: #ffffff !important;
        }

        ul[role="listbox"] {
            background-color: #ffffff !important;
        }

        li[role="option"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        div[data-baseweb="select"] {
            position: relative;
        }

        div[data-baseweb="select"] > div:first-child::after {
            content: "▼";
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            color: #0f172a;
            font-size: 1rem;
        }

        .stSlider > div > div {
            background-color: #e2e8f0 !important;
            height: 8px !important;
            border-radius: 4px !important;
            border: none !important;
            box-shadow: none !important;
        }

        .stSlider > div {
            background-color: transparent !important;
            border: none !important;
        }

        .stSlider > div > div > div {
            background-color: #dc2626 !important;
            height: 8px !important;
            border-radius: 4px !important;
        }

        .stSlider [role="slider"] {
            background-color: #dc2626 !important;
            width: 20px !important;
            height: 20px !important;
            border-radius: 50% !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }

        [data-testid="stFileUploader"] label {
            border: 2px solid #000000 !important;
            border-radius: 1rem !important;
            padding: 0.5rem 1rem !important;
            display: inline-block !important;
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
                dataset = st.radio(
                    "Select Dataset", 
                    ("DFDC", "FFPP"),
                    index=0 if st.session_state.dataset == "DFDC" else 1,
                    key="dataset_radio_key"
                )
                st.session_state.dataset = dataset
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
            result_text = data["result"]
            pred = float(data["pred"])
            
            # Format results properly
            is_fake = result_text == 'fake'
            confidence = pred * 100 if is_fake else (1 - pred) * 100
            percentage = round(confidence)
            
            st.markdown(f"""
                <div style="background-color: #ffffff; padding: 3rem; border-radius: 2rem; margin: 2rem 0;">
                    <h2 style="font-size: 4rem; font-weight: 900; color: {'#dc2626' if is_fake else '#16a34a'}; text-align: center; margin-bottom: 2rem;">
                        {'Deepfake Detected' if is_fake else 'Authentic Media'}
                    </h2>
                    <p style="font-size: 2.5rem; color: #0f172a; text-align: center; margin-bottom: 2rem;">
                        {'This media appears to be manipulated' if is_fake else 'No manipulation detected'}
                    </p>
                    <p style="font-size: 3rem; font-weight: 700; color: #0f172a; text-align: center;">
                        Confidence: {percentage}%
                    </p>
                </div>
            """, unsafe_allow_html=True)

            if st.button("ANALYZE ANOTHER FILE"):
                st.session_state.result_data = None
                st.rerun()
