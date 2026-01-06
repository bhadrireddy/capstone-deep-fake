import streamlit as st
from PIL import Image
import time

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. DUMMY BACKEND (DELETE THIS SECTION WHEN USING YOUR ACTUAL API) ---
# NOTE: Replace these functions with: from api import process_image, process_video
def process_image(image, model, dataset, threshold):
    time.sleep(2) # Simulate processing time
    # Return 'real' or 'fake' and a probability score (0.0 to 1.0)
    return "fake", 0.87 

def process_video(video_path, model, dataset, threshold, frames):
    time.sleep(3) # Simulate processing time
    return "real", 0.12
# -----------------------------------------------------------------------

# --- 3. PROFESSIONAL CSS STYLING ---
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f4f8 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .app-header {
        background: white;
        padding: 20px 40px;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 99;
    }
    .header-logo {
        font-size: 24px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563eb, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 120px 20px;
        max-width: 1000px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-title {
        font-size: 80px;
        font-weight: 900;
        color: #1e293b;
        margin-bottom: 24px;
        line-height: 1.1;
        letter-spacing: -1px;
        text-align: center;
    }
    .hero-subtitle {
        font-size: 36px;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 32px;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .hero-desc {
        font-size: 22px;
        color: #475569;
        margin-bottom: 50px;
        line-height: 1.7;
        text-align: center;
        font-weight: 400;
        max-width: 800px;
    }
    
    /* Upload Section */
    .upload-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        text-align: center;
        border: 2px dashed #cbd5e1;
        transition: border-color 0.3s;
        margin-bottom: 30px;
    }
    .upload-container:hover {
        border-color: #3b82f6;
    }
    .upload-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    
    /* Result Card */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .result-real {
        border-top: 8px solid #10b981;
    }
    .result-fake {
        border-top: 8px solid #ef4444;
    }
    .prediction-text {
        font-size: 36px;
        font-weight: 800;
        margin: 20px 0;
    }
    
    /* Custom Progress Bar */
    .custom-progress-bg {
        background-color: #e2e8f0;
        border-radius: 10px;
        height: 25px;
        width: 100%;
        overflow: hidden;
        margin-top: 10px;
    }
    .custom-progress-fill {
        height: 100%;
        text-align: right;
        padding-right: 10px;
        line-height: 25px;
        color: white;
        font-weight: bold;
        font-size: 14px;
        transition: width 1s ease-in-out;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    
    /* Get Started Button - Hero Page */
    .hero-container .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        font-size: 20px;
        font-weight: 700;
        padding: 20px 60px;
        height: auto;
        min-height: 64px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .hero-container .stButton > button:hover {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.4);
    }
    
    /* Info Box */
    .info-box {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 14px;
        color: #1e3a8a;
    }
    
    /* Step Styling for Modal */
    .step-card {
        background: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
    }
    .step-title {
        font-weight: 700;
        color: #0f172a;
    }

</style>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE INITIALIZATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'hero'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'pred_score' not in st.session_state:
    st.session_state.pred_score = 0.0
if 'show_modal' not in st.session_state:
    st.session_state.show_modal = False

# --- 5. HELPER FUNCTIONS ---
def reset_app():
    st.session_state.page = 'hero'
    st.session_state.uploaded_file = None
    st.session_state.result = None

def go_to_main():
    st.session_state.page = 'main'

def show_modal():
    st.session_state.show_modal = True

def close_modal():
    st.session_state.show_modal = False

# --- 6. PAGE: HERO SECTION ---
if st.session_state.page == 'hero':
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Deepfake Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Detect AI-generated images and videos</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-desc">Upload an image or video to check if it has been manipulated using deepfake techniques. Our advanced AI models analyze your media with state-of-the-art precision.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Get Started", use_container_width=True, type="primary"):
            go_to_main()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. MAIN & RESULT LOGIC ---
else:
    # -- HEADER --
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown('<div class="header-logo"> DF Detector</div>', unsafe_allow_html=True)
    with header_col2:
        if st.button("How It Works"):
            show_modal()

    # -- MODAL (HOW IT WORKS) --
    if st.session_state.show_modal:
        with st.expander("How It Works", expanded=True):
            st.markdown("""
            <div class="step-card"><span class="step-title">1. Upload:</span> Select an image or video file.</div>
            <div class="step-card"><span class="step-title">2. Analyze:</span> Our AI extracts frames and scans for artifacts.</div>
            <div class="step-card"><span class="step-title">3. Predict:</span> The model calculates a probability score.</div>
            """, unsafe_allow_html=True)
            if st.button("Close Help"):
                close_modal()
                st.rerun()

    # -- PAGE: MAIN (UPLOAD & SETTINGS) --
    if st.session_state.page == 'main':
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two column layout for upload and settings
        main_col1, main_col2 = st.columns([2, 1])
        
        with main_col1:
            st.markdown("### 📤 Upload Media")
            file_type = st.radio("File Type", ["Image", "Video"], horizontal=True, label_visibility="collapsed")
            
            # Dynamic File Uploader
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            if file_type == "Image":
                st.markdown('<div class="upload-icon"></div>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            else:
                st.markdown('<div class="upload-icon">🎥</div>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Choose a video...", type=['mp4'])
            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.success(f" {uploaded_file.name} uploaded successfully!")
                
                # Preview
                if file_type == "Image":
                    st.image(uploaded_file, caption="Preview", width=300)
                else:
                    st.video(uploaded_file)
            
        with main_col2:
            st.markdown("### ⚙️ Settings")
            with st.expander("Advanced Configuration", expanded=True):
                model_choice = st.selectbox("Select Model", ["EfficientNetB4", "EfficientNetAutoAttB4", "ResNext50"], index=1)
                st.markdown('<div class="info-box">AutoAttB4 is recommended for highest accuracy on facial artifacts.</div>', unsafe_allow_html=True)
                
                dataset_choice = st.radio("Dataset", ["DFDC (General)", "FFPP (FaceSwap)"])
                st.markdown('<div class="info-box">DFDC is better for general deepfakes.</div>', unsafe_allow_html=True)
                
                threshold = st.slider("Sensitivity Threshold", 0.1, 0.9, 0.5)
                st.markdown('<div class="info-box">Lower threshold = stricter detection.</div>', unsafe_allow_html=True)
                
                if file_type == "Video":
                    frames = st.slider("Frames to Analyze", 10, 100, 20)

        # Analyze Button
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🔍 Check for Deepfake", type="primary", use_container_width=True, disabled=(st.session_state.uploaded_file is None))
        
        if analyze_btn:
            with st.spinner("Analyzing media features... Please wait."):
                # Call the processing function (Mocked or Real)
                try:
                    if file_type == "Image":
                        result, score = process_image(st.session_state.uploaded_file, model_choice, dataset_choice, threshold)
                    else:
                        # For video, you normally need to save it to disk first
                        with open("temp_video.mp4", "wb") as f:
                            f.write(st.session_state.uploaded_file.getbuffer())
                        result, score = process_video("temp_video.mp4", model_choice, dataset_choice, threshold, frames if 'frames' in locals() else 20)
                    
                    st.session_state.result = result
                    st.session_state.pred_score = score
                    st.session_state.page = 'result'
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # -- PAGE: RESULT --
    elif st.session_state.page == 'result':
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Result Variables
        is_real = st.session_state.result == 'real'
        score = st.session_state.pred_score
        
        # Calculate percentages
        if is_real:
            confidence = (1 - score) * 100 # Assuming score is "fake" probability
            main_color = "#10b981" # Green
            icon = "✅"
            status = "AUTHENTIC"
            card_class = "result-real"
        else:
            confidence = score * 100
            main_color = "#ef4444" # Red
            icon = "🚨"
            status = "DEEPFAKE DETECTED"
            card_class = "result-fake"

        # Result Card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div style="font-size: 80px;">{icon}</div>
            <div class="prediction-text" style="color: {main_color};">{status}</div>
            <div style="font-size: 24px; color: #64748b;">Confidence Score</div>
            <div style="font-size: 56px; font-weight: 800; color: #1e293b;">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Visual Gauge / Progress Bar
        st.markdown("### Analysis Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Real Probability**")
            real_score = (1-score)*100
            st.markdown(f"""
            <div class="custom-progress-bg">
                <div class="custom-progress-fill" style="width: {real_score}%; background-color: #10b981;">{real_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Fake Probability**")
            fake_score = score*100
            st.markdown(f"""
            <div class="custom-progress-bg">
                <div class="custom-progress-fill" style="width: {fake_score}%; background-color: #ef4444;">{fake_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Action Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Analyze Another File", use_container_width=True):
                st.session_state.uploaded_file = None
                st.session_state.result = None
                st.session_state.page = 'main'
                st.rerun()
        with btn_col2:
            if st.button("Back to Home", use_container_width=True, type="secondary"):
                reset_app()
                st.rerun()
