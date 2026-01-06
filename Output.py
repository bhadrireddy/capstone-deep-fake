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
        
        /* Global Font & Text Visibility */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Force Text Colors to Dark Blue/Black for Visibility */
        h1, h2, h3, h4, h5, h6, span, div, label, p {
            color: #0f172a !important; /* Dark Slate 900 */
        }
        
        /* Background: Subtler Warm Light Color */
        .stApp {
            background: linear-gradient(180deg, #FFFBF0 0%, #FFF5E6 100%);
            background-attachment: fixed;
        }

        /* Hide standard Streamlit Header/Footer/Menu */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* --- HERO PAGE STYLES --- */
        .hero-title {
            font-size: 8rem !important; /* Massive Title */
            font-weight: 900 !important;
            line-height: 1.1;
            letter-spacing: -0.02em;
            color: #1e3a8a !important; /* Strong Blue for Heading */
            text-shadow: 2px 2px 0px rgba(255,255,255,0.5);
            margin-bottom: 1rem;
        }
        
        .hero-subtitle {
            font-size: 3.5rem !important; /* Massive Subtitle */
            font-weight: 700 !important;
            color: #334155 !important; /* Slate 700 */
            margin-bottom: 3rem;
        }

        /* --- WIDGET SIZING (HUGE) --- */
        
        /* Buttons (Start Detecting & Check) */
        .stButton button {
            background: linear-gradient(135deg, #e05252 0%, #c53030 100%) !important;
            color: white !important;
            border: none;
            padding: 3rem 2rem !important; /* Huge padding */
            border-radius: 2rem !important;
            font-weight: 900 !important;
            font-size: 3rem !important; /* Huge Font */
            line-height: 1 !important;
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .stButton button p {
            color: white !important; /* Force button text white */
            font-size: 3rem !important;
        }
        .stButton button:hover {
            transform: scale(1.02);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }
        
        /* ================= FILE UPLOADER ================= */

        /* Outer uploader container: bigger, white */
        [data-testid="stFileUploader"] {
            padding: 5rem 3rem !important;
            border-radius: 2.5rem !important;
            background-color: #ffffff !important;          /* white box */
            border: 3px dashed #64748b !important;
        }

/* Label */
[data-testid="stFileUploader"] label {
    font-size: 4rem !important;
    font-weight: 900 !important;
    color: #0f172a !important;
}

/* Kill dark dropzone wrapper */
[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    background-image: none !important;
    box-shadow: none !important;
    border-radius: 1.5rem !important;
}

/* Inner clickable area */
[data-testid="stFileUploader"] div[role="button"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    padding: 2.5rem 2rem !important;
    border-radius: 1.5rem !important;
}

/* Helper text */
[data-testid="stFileUploader"] small {
    color: #475569 !important;
    font-size: 1.8rem !important;
}


        /* Text inside drag-and-drop button */
        [data-testid="stFileUploader"] div[role="button"] p {
            color: #0f172a !important;
            font-size: 2.8rem !important;
            font-weight: 800 !important;
        }

        /* "Limit 200MB per file..." – inside same box & bigger */
        [data-testid="stFileUploader"] small {
             font-size: 1.8rem !important;
             display: block;
             margin-top: 0.8rem;
             color: #0f172a !important;                    /* dark, readable */
             font-weight: 600 !important;
        }

        /* Radio Buttons (File Type) - huge (Image / Video) */
        .stRadio > div {
            flex-direction: row;
            gap: 50px;
            justify-content: center;
        }
        .stRadio label {
            font-size: 25rem !important; /* bigger radio text */
            padding: 2rem 4rem !important;
            background-color: rgba(255,255,255,0.9);
            border: 3px solid #cbd5e1;
            border-radius: 2.2rem;
            cursor: pointer;
        }

        /* Select Box (Dropdowns) */
        .stSelectbox label {
            font-size: 8rem !important;  /* tremendously bigger label */
            margin-bottom: 1rem !important;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            font-size: 6.2rem !important;     /* bigger value text */
            min-height: 6rem !important;      /* taller dropdown */
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            color: #111827 !important;        /* force dark text */
        }
        .stSelectbox div[data-baseweb="select"] ul li {
            font-size: 2.2rem !important;
            color: #111827 !important;        /* dropdown option text black */
        }
        
        /* Sliders */
        .stSlider label {
            font-size: 3rem !important; /* bigger label */
        }
        .stSlider div[data-baseweb="slider"] {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stSlider span {
            font-size: 6.2rem !important; /* tick / value text larger */
        }
        
        /* Expander (Advanced Settings) */
        .streamlit-expanderHeader {
            font-size: 10.5rem !important;  /* INCREASED SIZE FOR ADVANCED SETTINGS LABEL */
            font-weight: 800 !important;
            background-color: rgba(255,255,255,0.9) !important; /* keep light, avoid black */
            border-radius: 1rem;
            padding: 2.2rem !important;
            color: #0f172a !important;
        }
        .streamlit-expanderContent {
            background-color: rgba(255,255,255,0.7) !important;
            padding-top: 1.8rem !important;
            padding-bottom: 1.8rem !important;
        }
        .streamlit-expanderContent > div {
            background-color: transparent !important; /* avoid inner white box */
            box-shadow: none !important;
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
                <p style="font-size: 2.2rem; font-weight: 500; color: #475569 !important; max-width: 70rem; margin: 0 auto 5rem auto; line-height: 1.6;">
                    Upload an image or video to check if it has been manipulated using deepfake techniques. 
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the start button
        c1, c2, c3 = st.columns([1, 4, 1]) # Wider middle column for massive button
        with c2:
            st.button("START DETECTING ➜", on_click=go_to_main, use_container_width=True)


# --- PAGE 2 & 3: MAIN UPLOAD & RESULTS ---
elif st.session_state.page == 'main' or st.session_state.page == 'results':
    
    # Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 4rem; margin-top: 2rem;">
            <h1 style="font-size: 5rem; font-weight: 900; color: #1e3a8a !important;">Deepfake Detector</h1>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 10, 1]) # Max width usage
    
    with col2:
        # --- INPUT SECTION ---
        if st.session_state.result_data is None:
            
            # File Type Radio - Huge
            st.markdown('<p style="font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem;">Select File Type</p>', unsafe_allow_html=True)
            file_type = st.radio(
                "Select file type:", 
                ("Image", "Video"), 
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            # File Uploader - Huge
            st.markdown('<p style="font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem;">Upload File</p>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                f"Drop your {file_type.lower()} here...", 
                type=["jpg", "jpeg", "png", "mp4"]
            )
            
            if uploaded_file is not None:
                st.markdown(f'<div style="font-size: 1.5rem; color: green; font-weight: bold; margin-bottom: 1rem;">File uploaded: {uploaded_file.name}</div>', unsafe_allow_html=True)
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

            # Advanced Settings - Huge
            with st.expander("Show Advanced Settings", expanded=False):
                # NOTE: sizing of expander label itself is controlled in CSS above
                st.markdown('<div style="padding: 2rem; background: rgba(255,255,255,0.0); border-radius: 1rem;">', unsafe_allow_html=True)
                
                # Using columns for better alignment of big inputs
                adv_c1, adv_c2 = st.columns(2)
                
                with adv_c1:
                    model = st.selectbox(
                        "Select Model", 
                        ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
                
                with adv_c2:
                    threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
                    st.markdown("<br>", unsafe_allow_html=True)
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
            
            # Action Buttons
            if st.button("ANALYZE ANOTHER FILE"):
                st.session_state.result_data = None
                st.rerun()

