import streamlit as st
from PIL import Image
from api import process_image, process_video

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hero Section */
    .hero-wrapper {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        margin: 0;
        width: 100%;
    }
    
    .hero-container {
        text-align: center;
        padding: 0 40px;
        width: 100%;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 72px;
        font-weight: 900;
        color: white;
        margin-bottom: 24px;
        line-height: 1.1;
        letter-spacing: -1px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 32px;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 600;
        margin-bottom: 32px;
        letter-spacing: -0.3px;
    }
    
    .hero-desc {
        font-size: 20px;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 48px;
        line-height: 1.7;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Main Page Container */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 48px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 20px auto;
    }
    
    /* Large Text Styles */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    
    .stMarkdown h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #1e293b !important;
        margin-bottom: 24px;
    }
    
    .stMarkdown h2 {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #334155 !important;
    }
    
    .stMarkdown h3 {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    
    /* Large Buttons */
    .stButton > button {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 16px 40px !important;
        height: auto !important;
        min-height: 56px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    /* Hero Start Button */
    .hero-wrapper .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 20px 60px !important;
        min-height: 68px !important;
        box-shadow: 0 8px 20px rgba(220, 38, 38, 0.4) !important;
        border: none !important;
    }
    
    .hero-wrapper .stButton > button:hover {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px rgba(220, 38, 38, 0.5) !important;
    }
    
    /* Primary Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 16px 40px !important;
        min-height: 56px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Form Elements - Larger */
    .stRadio > div {
        font-size: 18px !important;
    }
    
    .stRadio > div > label {
        font-size: 18px !important;
        padding: 12px 20px !important;
    }
    
    .stSelectbox label,
    .stSlider label {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 12px !important;
    }
    
    .stSelectbox > div > div {
        font-size: 16px !important;
    }
    
    .stSlider {
        margin-top: 20px !important;
    }
    
    .stFileUploader {
        font-size: 16px !important;
    }
    
    /* Info Messages */
    .stInfo {
        font-size: 16px !important;
        padding: 16px 20px !important;
    }
    
    /* Result Styling */
    .result-text {
        font-size: 28px !important;
        font-weight: 700 !important;
        margin: 20px 0 !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 48px;
        }
        .hero-subtitle {
            font-size: 24px;
        }
        .hero-desc {
            font-size: 18px;
        }
        .main-container {
            padding: 32px 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'hero'

# Hero Page
if st.session_state.page == 'hero':
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="hero-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="hero-title">Deepfake Detector</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="hero-subtitle">Detect AI-generated images and videos</h2>', unsafe_allow_html=True)
    st.markdown('<p class="hero-desc">Upload an image or video to check if it has been manipulated using deepfake techniques.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start", use_container_width=True, type="primary", key="start_btn"):
            st.session_state.page = 'main'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Page
else:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Set the title of your Streamlit app
    st.title("Deepfake Detector App")
    
    # Choose between image and video upload
    file_type = st.radio("Select file type:", ("Image", "Video"))
    
    # Upload file through Streamlit
    uploaded_file = st.file_uploader(f"Choose a {file_type.lower()}...", type=[
        "jpg", "jpeg", "png", "mp4"])
    
    model = st.selectbox("Select Model", ("EfficientNetB4", "EfficientNetB4ST",
                         "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST"))
    dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
    threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
    
    if file_type == "Video":
        frames = st.slider("Select Frames", 0, 100, 50)
    
    # Display the uploaded file
    if uploaded_file is not None:
        if file_type == "Image":
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=200)
            except Exception as e:
                print(e)
                st.error(f"Error: Invalid Filetype")
        else:
            st.video(uploaded_file)
    
        # Check if the user wants to perform the deepfake detection
        if st.button("Check for Deepfake", type="primary", use_container_width=True):
            # Convert file to bytes for API request
            if file_type == "Image":
                # uploaded_file = check_and_convert_image(uploaded_file)
                result, pred = process_image(
                    image=uploaded_file, model=model, dataset=dataset, threshold=threshold)
                st.markdown(
                    f'''
                    <style>
                        .result{{
                            color: {'#ff4b4b' if result == 'fake' else '#6eb52f'};
                        }}
                    </style>
                    <h3 class="result-text">The given {file_type} is: <span class="result"> {result} </span> with a probability of <span class="result">{pred:.2f}</span></h3>''', unsafe_allow_html=True)
    
            else:
                with open(f"uploads/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.read())
    
                video_path = f"uploads/{uploaded_file.name}"
    
                result, pred = process_video(video_path, model=model,
                                             dataset=dataset, threshold=threshold, frames=frames)
    
                st.markdown(
                     f'''
                    <style>
                        .result{{
                            color: {'#ff4b4b' if result == 'fake' else '#6eb52f'};
                        }}
                    </style>
                    <h3 class="result-text">The given {file_type} is: <span class="result"> {result} </span> with a probability of <span class="result">{pred:.2f}</span></h3>''', unsafe_allow_html=True)
    else:
        st.info("Please upload a file.")
    
    st.markdown('</div>', unsafe_allow_html=True)
