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

        /* --- FIXED RADIO BUTTON CSS STARTS HERE --- */

        /* 1. COMPLETELY HIDE the default Streamlit visual circles (the source of the double dot) */
        .stRadio [data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
        
        /* 2. Style the NATIVE INPUT to replace the hidden default style */
        .stRadio input[type="radio"] {
            display: inline-block !important;
            opacity: 1 !important;
            visibility: visible !important;
            
            /* Remove default browser styling (removes the black dot) */
            appearance: none !important;
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            
            /* Custom Circle Styling */
            width: 20px !important;
            height: 20px !important;
            margin-right: 0.75rem !important;
            
            /* Force White Background for unselected state */
            background-color: #ffffff !important;
            border: 2px solid #000000 !important;
            border-radius: 50% !important;
            
            outline: none !important;
            position: relative !important;
            z-index: 11 !important;
            cursor: pointer !important;
        }

        /* 3. Styling for when the button is CHECKED */
        .stRadio input[type="radio"]:checked {
            background-color: #ffffff !important; /* Keep background white */
            border-color: #dc2626 !important;     /* Turn border red */
        }

        /* 4. The inner red dot for the checked state */
        .stRadio input[type="radio"]:checked::after {
            content: "" !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: 8px !important;
            height: 8px !important;
            border-radius: 50% !important;
            background-color: #dc2626 !important;
            display: block !important;
        }
        
        /* --- FIXED RADIO BUTTON CSS ENDS HERE --- */

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

            # Restrict allowed file types based on selected file type
            if file_type == "Image":
                upload_types = ["jpg", "jpeg", "png"]
            else:
                upload_types = ["mp4"]

            uploaded_file = st.file_uploader(
                f"Drop your {file_type.lower()} here...",
                type=upload_types
            )

            # Display uploaded file preview right under the uploader (small size)
            if uploaded_file:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if file_type == "Image":
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, width=300)
                    else:
                        # For videos, show video player
                        video_bytes = uploaded_file.read()
                        st.video(video_bytes)
                        # Reset file pointer for processing
                        uploaded_file.seek(0)

            # ✅ FIX: always define frames
            frames = 0

            with st.expander("Show Advanced Settings"):
                model = st.selectbox(
                    "Select Model",
                    (
                        "EfficientNetB4",
                        "EfficientNetB4ST",
                        "EfficientNetAutoAttB4",
                        "EfficientNetAutoAttB4ST",
                    )
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
                    "file_type": file_type,
                    "file_name": uploaded_file.name
                }
                st.rerun()

        else:
            data = st.session_state.result_data
            result_text = data["result"]
            pred = float(data["pred"])
            
            # Clamp prediction to [0, 1] range to prevent > 100% confidence
            pred = max(0.0, min(1.0, pred))
            
            # Format results properly - Three-class system: Fake, Suspicious, Likely Real
            result_lower = result_text.lower()
            is_fake = result_lower == 'fake'
            is_suspicious = result_lower == 'suspicious'
            is_likely_real = result_lower == 'likely real'
            
            # Calculate confidence percentage based on prediction value
            # For Fake: high confidence (70-100%)
            # For Suspicious: moderate confidence (50-70%)
            # For Likely Real: lower confidence (30-60%) to reflect uncertainty
            if is_fake:
                # Fake: map pred [0.65, 1.0] to confidence [70%, 100%]
                if pred >= 0.65:
                    distance = (pred - 0.65) / (1.0 - 0.65)
                    confidence = 70 + (distance * 30)  # 70% to 100%
                else:
                    confidence = 65
                color = '#dc2626'  # Red
                title = 'Deepfake Detected'
                message = 'This media appears to be manipulated or AI-generated'
            elif is_suspicious:
                # Suspicious: map pred [0.35, 0.65] to confidence [50%, 70%]
                if pred >= 0.35 and pred <= 0.65:
                    distance = (pred - 0.35) / (0.65 - 0.35)
                    confidence = 50 + (distance * 20)  # 50% to 70%
                else:
                    confidence = 60
                color = '#f59e0b'  # Amber/Orange
                title = 'Suspicious Content'
                message = 'Uncertain result - this media may be manipulated. Please verify with additional analysis.'
            else:  # Likely Real
                # Likely Real: map pred [0.0, 0.35] to confidence [30%, 60%]
                # Lower confidence for real to prevent false negatives
                if pred <= 0.35:
                    if pred > 0:
                        distance = (0.35 - pred) / 0.35
                        confidence = 30 + (distance * 30)  # 30% to 60%
                    else:
                        confidence = 30
                else:
                    confidence = 35
                color = '#16a34a'  # Green
                title = 'Likely Authentic'
                message = 'No significant manipulation detected, but confidence is moderate'
            
            # Ensure confidence is between 0-100%
            confidence = max(0.0, min(100.0, confidence))
            percentage = round(confidence)
            
            st.markdown(f"""
                <div style="background-color: #ffffff; padding: 3rem; border-radius: 2rem; margin: 2rem 0; border: 4px solid {color};">
                    <h2 style="font-size: 4rem; font-weight: 900; color: {color}; text-align: center; margin-bottom: 2rem;">
                        {title}
                    </h2>
                    <p style="font-size: 2.5rem; color: #0f172a; text-align: center; margin-bottom: 2rem;">
                        {message}
                    </p>
                    <p style="font-size: 3rem; font-weight: 700; color: #0f172a; text-align: center;">
                        Confidence: {percentage}%
                    </p>
                    <p style="font-size: 1.5rem; color: #64748b; text-align: center; margin-top: 1rem;">
                        Prediction Score: {pred:.3f}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            if st.button("ANALYZE ANOTHER FILE"):
                st.session_state.result_data = None
                st.rerun()
