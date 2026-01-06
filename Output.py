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
        
        /* Global Font */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #2d3748; /* Dark gray text for light background */
        }
        
        /* 2) Background: Subtler Warm Light Color */
        .stApp {
            background: linear-gradient(180deg, #FFFBF0 0%, #FFF5E6 100%);
            background-attachment: fixed;
        }

        /* Hide standard Streamlit Header/Footer/Menu */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 1) Hero Title - Huge */
        .hero-title {
            font-size: 7rem;
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -0.02em;
            color: #1a202c;
            margin-bottom: 1rem;
        }
        
        /* 1) Hero Subtitle - Huge */
        .hero-subtitle {
            font-size: 3rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 3rem;
        }

        /* 6) Huge Buttons */
        .stButton button {
            background: linear-gradient(135deg, #e05252 0%, #c53030 100%);
            color: white;
            border: none;
            padding: 2rem 4rem; /* Huge padding */
            border-radius: 1.5rem;
            font-weight: 800;
            font-size: 2rem !important; /* Huge text */
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }
        
        /* 6) Huge File Uploader */
        [data-testid="stFileUploader"] {
            background-color: rgba(255,255,255,0.6);
            border: 4px dashed #cbd5e1;
            border-radius: 2rem;
            padding: 5rem; /* Huge drop area */
            text-align: center;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"] section {
            padding: 0;
        }
        [data-testid="stFileUploader"] button {
            display: none; /* Hide the small default browse button if possible or style it */
        }
        /* Make the drag drop text larger */
        [data-testid="stFileUploader"] div div {
            font-size: 1.5rem !important;
        }

        /* 6) Huge Radio Buttons */
        .stRadio > div {
            flex-direction: row;
            gap: 40px;
            justify-content: center;
        }
        .stRadio label {
            font-size: 2.5rem !important; /* Huge text */
            padding: 1rem 2rem;
            background-color: rgba(255,255,255,0.5);
            border-radius: 1rem;
            cursor: pointer;
        }
        
        /* 7) Align Advanced Settings & Make Selectbox/Sliders bigger */
        .stSelectbox label, .stSlider label {
            font-size: 1.5rem !important;
            font-weight: 600;
        }
        .stSelectbox div[data-baseweb="select"] {
            font-size: 1.25rem;
            height: 4rem;
        }
        
        /* Hide images in fullscreen */
        button[title="View fullscreen"]{
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions for Navigation ---
def go_to_main():
    st.session_state.page = 'main'

def go_to_hero():
    st.session_state.page = 'hero'
    st.session_state.result_data = None

# --- PAGE 1: HERO SECTION ---
if st.session_state.page == 'hero':
    # Using columns to center everything perfectly
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding-top: 15vh; margin-bottom: 2rem;">
                <h1 class="hero-title">Deepfake<br>Detector</h1>
                <h2 class="hero-subtitle">
                    Detect AI-generated images and videos
                </h2>
                <p style="font-size: 1.8rem; color: #718096; max-width: 60rem; margin: 0 auto 4rem auto; line-height: 1.6;">
                    Upload an image or video to check if it has been manipulated using deepfake techniques. 
                    High accuracy AI analysis at your fingertips.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the start button
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.button("START DETECTION ➜", on_click=go_to_main, use_container_width=True)


# --- PAGE 2 & 3: MAIN UPLOAD & RESULTS ---
elif st.session_state.page == 'main' or st.session_state.page == 'results':
    
    # 3) Removed the white box wrapper. Elements float directly on the warm background.
    
    # Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem; margin-top: 2rem;">
            <h1 style="font-size: 4rem; font-weight: 900; color: #1a202c;">Deepfake Detector</h1>
            </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 8, 1]) # Keep content somewhat centered but wide
    
    with col2:
        # --- INPUT SECTION ---
        if st.session_state.result_data is None:
            
            # File Type Radio - Huge due to CSS
            st.markdown('<p style="font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 1rem;">Select File Type</p>', unsafe_allow_html=True)
            file_type = st.radio(
                "Select file type:", 
                ("Image", "Video"), 
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            # File Uploader - Huge due to CSS
            uploaded_file = st.file_uploader(
                f"Drop your {file_type.lower()} here or click to browse", 
                type=["jpg", "jpeg", "png", "mp4"]
            )
            
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                # Preview Media
                if file_type == "Image":
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Preview", use_container_width=True)
                    except Exception as e:
                        st.error("Error: Invalid Image File")
                else:
                    st.video(uploaded_file)

            st.markdown('<br>', unsafe_allow_html=True)

            # 7) Advanced Settings - Aligned properly
            with st.expander("Show Advanced Settings", expanded=False):
                st.markdown('<div style="padding: 2rem; background: rgba(255,255,255,0.5); border-radius: 1rem;">', unsafe_allow_html=True)
                
                # Using columns for better alignment of big inputs
                adv_c1, adv_c2 = st.columns(2)
                
                with adv_c1:
                    model = st.selectbox(
                        "Select Model", 
                        ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
                    )
                    dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
                
                with adv_c2:
                    threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
                    if file_type == "Video":
                        frames = st.slider("Select Frames", 0, 100, 50)
                    else:
                        frames = 0
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Check Button
            if uploaded_file is not None:
                st.markdown("<br><br>", unsafe_allow_html=True)
                # Huge button due to CSS
                if st.button("CHECK FOR DEEPFAKE", use_container_width=True):
                    with st.spinner("Analyzing media..."):
                        time.sleep(1) 
                        
                        if file_type == "Image":
                            result, pred = process_image(
                                image=uploaded_file, model=model, dataset=dataset, threshold=threshold)
                        else:
                            with open(f"uploads/{uploaded_file.name}", "wb") as f:
                                f.write(uploaded_file.read())
                            video_path = f"uploads/{uploaded_file.name}"
                            result, pred = process_video(
                                video_path, model=model, dataset=dataset, threshold=threshold, frames=frames)
                        
                        st.session_state.result_data = {
                            "result": result,
                            "pred": pred,
                            "file_type": file_type
                        }
                        st.rerun()

        # --- RESULTS SECTION ---
        else:
            # 9) Removed white box (card) wrapper. Floating directly on background.
            data = st.session_state.result_data
            result_text = data["result"]
            probability = data["pred"]
            
            if result_text == 'fake':
                color_class = "text-red-600"
                icon = "fa-exclamation-triangle"
                title = "DEEPFAKE DETECTED"
                desc = "This media appears to be manipulated."
                bar_color = "#e53e3e"
            else:
                color_class = "text-green-600"
                icon = "fa-check-circle"
                title = "AUTHENTIC MEDIA"
                desc = "No manipulation detected."
                bar_color = "#38a169"

            # Huge Result Display
            st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <i class="fas {icon} {color_class}" style="font-size: 8rem; margin-bottom: 1rem;"></i>
                    <h2 class="{color_class}" style="font-size: 4rem; font-weight: 900; margin-bottom: 0.5rem;">{title}</h2>
                    <p style="font-size: 2rem; color: #4a5568; margin-bottom: 2rem;">{desc}</p>
                    
                    <div style="font-size: 5rem; font-weight: 800; color: #2d3748; margin-bottom: 1rem;">
                        {probability:.2f}
                        <span style="font-size: 1.5rem; color: #718096; font-weight: 400; display: block;">Confidence Score</span>
                    </div>

                    <div style="width: 100%; background-color: #e2e8f0; height: 2rem; border-radius: 1rem; margin-top: 2rem; overflow: hidden;">
                        <div style="width: {probability*100}%; background-color: {bar_color}; height: 100%; border-radius: 1rem;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Action Buttons (8: Removed Download Report)
            if st.button("ANALYZE ANOTHER FILE"):
                st.session_state.result_data = None
                st.rerun()

# 5) Removed Footer completely
