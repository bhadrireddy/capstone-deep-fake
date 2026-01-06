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
        padding: 40px 20px;
        margin: 0;
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .hero-container {
        text-align: center;
        padding: 60px 40px;
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Increased hero title size and improved spacing */
    .hero-title {
        font-size: 84px;
        font-weight: 900;
        color: white;
        margin-bottom: 32px;
        line-height: 1.1;
        letter-spacing: -1.5px;
        text-shadow: 3px 3px 12px rgba(0,0,0,0.25);
    }
    
    /* Increased hero subtitle size */
    .hero-subtitle {
        font-size: 36px;
        color: rgba(255, 255, 255, 0.98);
        font-weight: 600;
        margin-bottom: 40px;
        letter-spacing: -0.3px;
    }
    
    /* Improved hero description styling */
    .hero-desc {
        font-size: 22px;
        color: rgba(255, 255, 255, 0.92);
        margin-bottom: 60px;
        line-height: 1.8;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        font-weight: 500;
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
    
    /* Increased hero button size and improved styling */
    .hero-wrapper .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        padding: 24px 80px !important;
        min-height: 76px !important;
        box-shadow: 0 10px 24px rgba(220, 38, 38, 0.45) !important;
        border: none !important;
        cursor: pointer !important;
        letter-spacing: 0.5px !important;
    }
    
    .hero-wrapper .stButton > button:hover {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 32px rgba(220, 38, 38, 0.55) !important;
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
    
    /* Improved responsive design for smaller screens */
    @media (max-width: 1024px) {
        .hero-title {
            font-size: 64px;
        }
        .hero-subtitle {
            font-size: 32px;
        }
        .hero-desc {
            font-size: 20px;
        }
        .hero-wrapper .stButton > button {
            font-size: 22px !important;
            padding: 20px 60px !important;
            min-height: 68px !important;
        }
    }
    
    @media (max-width: 768px) {
        .hero-container {
            padding: 48px 28px;
        }
        .hero-title {
            font-size: 48px;
            margin-bottom: 24px;
        }
        .hero-subtitle {
            font-size: 26px;
            margin-bottom: 28px;
        }
        .hero-desc {
            font-size: 18px;
            margin-bottom: 40px;
        }
        .hero-wrapper .stButton > button {
            font-size: 20px !important;
            padding: 18px 50px !important;
            min-height: 60px !important;
        }
        .main-container {
            padding: 32px 24px;
        }
    }
    
    @media (max-width: 480px) {
        .hero-container {
            padding: 40px 20px;
        }
        .hero-title {
            font-size: 36px;
            margin-bottom: 20px;
        }
        .hero-subtitle {
            font-size: 22px;
            margin-bottom: 20px;
        }
        .hero-desc {
            font-size: 16px;
            margin-bottom: 32px;
        }
        .hero-wrapper .stButton > button {
            font-size: 18px !important;
            padding: 16px 40px !important;
            min-height: 56px !important;
        }
    }
</style>
""", unsafe_allow_html=True)
