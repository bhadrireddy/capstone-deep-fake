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
        }
        
        /* Background: Warm Light Color */
        .stApp {
            background: linear-gradient(180deg, #FFFBF0 0%, #FFF5E6 100%);
            background-attachment: fixed;
        }

        /* Hide standard Streamlit Header/Footer/Menu */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* --- HERO PAGE --- */
        .hero-title {
            font-size: 8rem !important;
            font-weight: 900 !important;
            line-height: 1.1;
            color: #1e3a8a !important; /* Strong Blue */
            text-shadow: 2px 2px 0px rgba(255,255,255,0.5);
            margin-bottom: 1rem;
        }
        
        .hero-subtitle {
            font-size: 3.5rem !important;
            font-weight: 700 !important;
            color: #334155 !important;
            margin-bottom: 3rem;
        }

        /* --- WIDGET SIZING & COLORS --- */
        
        /* 1) HUGE Radio Buttons (Image/Video) */
        .stRadio > div {
            flex-direction: row;
            gap: 60px;
            justify-content: center;
        }
        .stRadio label {
            font-size: 4rem !important; /* Huge text */
            padding: 1.5rem 4rem !important; /* Huge clickable area */
            background-color: white;
            border: 4px solid #cbd5e1;
            border-radius: 2rem;
            cursor: pointer;
            color: #1e293b !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        /* 4) HUGE File Uploader Box */
        [data-testid="stFileUploader"] {
            padding: 6rem 2rem !important; /* Increased padding */
            background-color: rgba(255,255,255,0.8);
            border: 6px dashed #94a3b8;
            border-radius: 3rem;
            min-height: 400px; /* Force minimum height */
            align-content: center;
        }
        
        /* 2) Increase font of "Drop your image here" */
        [data-testid="stFileUploader"] div div {
            font-size: 3rem !important;
            font-weight: 700 !important;
            color: #334155 !important;
        }
        
        /* 3) Change color of "Drag and drop files here" subtext */
        [data-testid="stFileUploader"] small {
             font-size: 2rem !important;
             color: #ef4444 !important; /* Changed to Red for visibility */
             font-weight: 600 !important;
             display: block;
             margin-top: 1.5rem;
        }
        
        /* 5) Tremendous Font for Advanced Settings Header */
        .streamlit-expanderHeader {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            color: #1e3a8a !important;
            background-color: transparent !important;
            border: none !important;
        }
        .streamlit-expanderHeader p {
             font-size: 3.5rem !important;
             color: #1e3a8a !important;
        }
        
        /* 6) Fix Advanced Settings Black Background */
        /* We force the expander content to be transparent or light */
        [data-testid="stExpanderDetails"] {
            background-color: transparent !important;
            border: none !important;
            color: #0f172a !important;
        }
        
        /* 8) Dropdown Text Color & 10) Size for Select Model */
        .stSelectbox label {
            font-size: 2.5rem !important;
            color: #1e3a8a !important;
            margin-bottom: 1rem !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            font-size: 2rem !important;
            background-color: white !important;
            color: #0f172a !important; /* Force dark text */
            border-radius: 1rem;
            border: 2px solid #cbd5e1;
            min-height: 6rem !important; /* Taller box */
        }
        /* The text inside the selected option */
        .stSelectbox div[data-baseweb="select"] div {
            color: #0f172a !important; 
            font-weight: 600;
        }

        /* 9) Increase Size for Select Threshold & Frames */
        .stSlider label {
            font-size: 2.5rem !important;
            color: #1e3a8a !important;
            padding-bottom: 1rem;
        }
        .stSlider div[data-baseweb="slider"] {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* 10) Size for DFDC/FFPP Radio (Inside Expander) */
        /* Since we targeted global radio above, we just need to ensure these fit. 
           The global .stRadio style applies here too, making them huge automatically. */

        /* Buttons (Start Detecting & Check) */
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
            font-size: 3rem !important;
        }
        
        /* Hide images in fullscreen */
        button[title="View fullscreen"]{ display: none; }
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
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding-top: 15vh; margin-bottom: 2rem;">
                <h1 class="hero-title">Deepfake<br>Detector</h1>
                <h2 class="hero-subtitle">
                    Detect AI-generated images and videos
                </h2>
                <p style="font-size: 2.5rem; font-weight: 500; color: #475569 !important; max-width: 70rem; margin: 0 auto 5rem auto; line-height: 1.6;">
                    Upload an image or video to check if it has been manipulated using deepfake techniques. 
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 4, 1])
        with c2:
            st.button("START DETECTING ➜", on_click=go_to_main, use_container_width=True)


# --- PAGE 2 & 3: MAIN UPLOAD & RESULTS ---
elif st.session_state.page == 'main' or st.session_state.page == 'results':
    
    # Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 4rem; margin-top: 2rem;">
            <h1 style="font-size: 6rem; font-weight: 900; color: #1e3a8a !important;">Deepfake Detector</h1>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        # --- INPUT SECTION ---
        if st.session_state.result_data is None:
            
            # File Type Radio - HUGE
            st.markdown('<p style="font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 2rem; color: #0f172a;">Select File Type</p>', unsafe_allow_html=True)
            file_type = st.radio(
                "Select file type:", 
                ("Image", "Video"), 
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown('<br><br>', unsafe_allow_html=True)
            
            # File Uploader - HUGE
            uploaded_file = st.file_uploader(
                f"Drop your {file_type.lower()} here", 
                type=["jpg", "jpeg", "png", "mp4"]
            )
            
            if uploaded_file is not None:
                st.markdown(f'<div style="font-size: 2rem; color: green; font-weight: bold; margin-bottom: 1rem; text-align: center;">File uploaded: {uploaded_file.name}</div>', unsafe_allow_html=True)
                if file_type == "Image":
                    try:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Preview", use_container_width=True)
                    except Exception as e:
                        st.error("Error: Invalid Image File")
                else:
                    st.video(uploaded_file)

            st.markdown('<br>', unsafe_allow_html=True)

            # Advanced Settings
            with st.expander("Show Advanced Settings", expanded=False):
                # 7) Removed the markdown 'div' wrapper that was creating the empty white box.
                # Now the controls sit directly on the expander background (transparent/page bg).
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                adv_c1, adv_c2 = st.columns(2)
                
                with adv_c1:
                    model = st.selectbox(
                        "Select Model", 
                        ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    # This radio will inherit the HUGE style from above
                    st.markdown('<p style="font-size: 2.5rem; font-weight: 700; color: #1e3a8a; margin-bottom: 1rem;">Select Dataset</p>', unsafe_allow_html=True)
                    dataset = st.radio("Select Dataset", ("DFDC", "FFPP"), label_visibility="collapsed")
                
                with adv_c2:
                    threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
                    st.markdown("<br>", unsafe_allow_html=True)
                    if file_type == "Video":
                        frames = st.slider("Select Frames", 0, 100, 50)
                    else:
                        frames = 0
                

            # Check Button
            if uploaded_file is not None:
                st.markdown("<br><br>", unsafe_allow_html=True)
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

            st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <i class="fas {icon} {color_class}" style="font-size: 10rem; margin-bottom: 2rem;"></i>
                    <h2 class="{color_class}" style="font-size: 6rem; font-weight: 900; margin-bottom: 1rem;">{title}</h2>
                    <p style="font-size: 3rem; color: #1e293b !important; margin-bottom: 3rem;">{desc}</p>
                    
                    <div style="font-size: 7rem; font-weight: 800; color: #0f172a !important; margin-bottom: 1rem;">
                        {probability:.2f}
                        <span style="font-size: 2rem; color: #475569 !important; font-weight: 500; display: block;">Confidence Score</span>
                    </div>

                    <div style="width: 100%; background-color: #cbd5e1; height: 3rem; border-radius: 1.5rem; margin-top: 3rem; overflow: hidden;">
                        <div style="width: {probability*100}%; background-color: {bar_color}; height: 100%; border-radius: 1.5rem;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            if st.button("ANALYZE ANOTHER FILE"):
                st.session_state.result_data = None
                st.rerun()
