import streamlit as st
from PIL import Image
from api import process_image, process_video
import os

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Read the exact CSS and structure from frontend.html
with open('frontend.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Extract CSS from the HTML
import re
css_match = re.search(r'<style>(.*?)</style>', html_content, re.DOTALL)
css_content = css_match.group(1) if css_match else ""

# Extract JavaScript from the HTML  
js_match = re.search(r'<script>(.*?)</script>', html_content, re.DOTALL)
js_content = js_match.group(1) if js_match else ""

# Inject CSS into Streamlit
st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .main .block-container {{
        padding: 0;
        max-width: 100%;
    }}
    {css_content}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'hero'
if 'file_type' not in st.session_state:
    st.session_state.file_type = 'Image'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'model' not in st.session_state:
    st.session_state.model = 'EfficientNetAutoAttB4'
if 'dataset' not in st.session_state:
    st.session_state.dataset = 'DFDC'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'frames' not in st.session_state:
    st.session_state.frames = 50
if 'result' not in st.session_state:
    st.session_state.result = None
if 'pred' not in st.session_state:
    st.session_state.pred = None

# Hero Page
if st.session_state.page == 'hero':
    st.markdown("""
    <div class="gradient-bg" style="min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem;">
        <div class="text-center" style="max-width: 64rem; margin: 0 auto;">
            <h1 class="hero-title text-white mb-6">Deepfake Detector</h1>
            <h2 style="font-size: 2.5rem; font-weight: 600; color: white; margin-bottom: 2rem;">Detect AI-generated images and videos</h2>
            <p style="font-size: 1.25rem; color: rgba(255,255,255,0.9); margin-bottom: 3rem; max-width: 42rem; margin-left: auto; margin-right: auto; line-height: 1.75;">
                Upload an image or video to check if it has been manipulated using deepfake techniques. Our advanced AI models analyze media files to detect artificial manipulation with high accuracy.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Detection →", use_container_width=True, key="start_btn"):
            st.session_state.page = 'main'
            st.rerun()
    
    st.markdown("""
    <style>
    button[key="start_btn"] {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
        font-size: 1.25rem !important;
        font-weight: bold !important;
        padding: 1.25rem 4rem !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.4) !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Page
elif st.session_state.page == 'main':
    st.markdown("""
    <div class="gradient-bg" style="min-height: 100vh; padding: 3rem 1.5rem;">
        <div style="max-width: 80rem; margin: 0 auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2.5rem; font-weight: bold; color: white;">Deepfake Detector</h1>
                <button onclick="document.getElementById('howItWorksPanel').classList.add('open'); document.getElementById('overlay').classList.add('active');" 
                        style="background: rgba(255,255,255,0.2); color: white; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-weight: 600; border: none; cursor: pointer;">
                    <i class="fas fa-question-circle"></i> How It Works
                </button>
            </div>
            
            <div class="bg-white rounded-2xl card-shadow" style="padding: 2.5rem;">
    """, unsafe_allow_html=True)
    
    # How It Works Panel
    st.markdown("""
    <div class="overlay" id="overlay" onclick="document.getElementById('howItWorksPanel').classList.remove('open'); document.getElementById('overlay').classList.remove('active');"></div>
    <div class="how-it-works-panel" id="howItWorksPanel">
        <div style="padding: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.5rem; font-weight: bold; color: #111827;">How It Works</h2>
                <button onclick="document.getElementById('howItWorksPanel').classList.remove('open'); document.getElementById('overlay').classList.remove('active');" 
                        style="background: none; border: none; font-size: 1.5rem; color: #6b7280; cursor: pointer;">×</button>
            </div>
            <div style="display: flex; flex-direction: column; gap: 1.5rem;">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="flex-shrink: 0; width: 3rem; height: 3rem; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.25rem;">1</div>
                    <div>
                        <h3 style="font-size: 1.125rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Upload Your Media</h3>
                        <p style="color: #4b5563;">Select an image or video file from your device. We support JPG, PNG, and MP4 formats.</p>
                    </div>
                </div>
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="flex-shrink: 0; width: 3rem; height: 3rem; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.25rem;">2</div>
                    <div>
                        <h3 style="font-size: 1.125rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Analyze Frames & Features</h3>
                        <p style="color: #4b5563;">Our AI model examines facial features, artifacts, and inconsistencies that indicate manipulation.</p>
                    </div>
                </div>
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="flex-shrink: 0; width: 3rem; height: 3rem; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 1.25rem;">3</div>
                    <div>
                        <h3 style="font-size: 1.125rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Get Instant Results</h3>
                        <p style="color: #4b5563;">Receive a detailed analysis with confidence scores showing whether the media is authentic or manipulated.</p>
                    </div>
                </div>
            </div>
            <div style="margin-top: 2rem; padding: 1rem; background: #eff6ff; border-radius: 0.5rem;">
                <h4 style="font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Advanced Settings</h4>
                <p style="font-size: 0.875rem; color: #4b5563;">Fine-tune detection with custom models, datasets, and threshold values.</p>
            </div>
        </div>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)
    
    # File Type Selection
    st.markdown('<label style="display: block; font-size: 1.5rem; font-weight: bold; color: #111827; margin-bottom: 1rem;">Select File Type</label>', unsafe_allow_html=True)
    
    file_type_col1, file_type_col2 = st.columns(2)
    with file_type_col1:
        image_active = st.session_state.file_type == 'Image'
        if st.button("📷 Image", key="image_btn", use_container_width=True):
            st.session_state.file_type = 'Image'
            st.session_state.uploaded_file = None
            st.rerun()
    with file_type_col2:
        video_active = st.session_state.file_type == 'Video'
        if st.button("🎥 Video", key="video_btn", use_container_width=True):
            st.session_state.file_type = 'Video'
            st.session_state.uploaded_file = None
            st.rerun()
    
    # Style file type buttons
    st.markdown(f"""
    <style>
    button[key="image_btn"] {{
        border: 3px solid {'#667eea' if image_active else '#e5e7eb'} !important;
        background: {'#f3f4ff' if image_active else 'white'} !important;
        color: {'#667eea' if image_active else '#374151'} !important;
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
    }}
    button[key="video_btn"] {{
        border: 3px solid {'#667eea' if video_active else '#e5e7eb'} !important;
        background: {'#f3f4ff' if video_active else 'white'} !important;
        color: {'#667eea' if video_active else '#374151'} !important;
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # File Upload
    st.markdown('<label style="display: block; font-size: 1.5rem; font-weight: bold; color: #111827; margin-bottom: 1rem; margin-top: 2rem;">Upload File</label>', unsafe_allow_html=True)
    
    file_types = ["jpg", "jpeg", "png"] if st.session_state.file_type == 'Image' else ["mp4"]
    uploaded_file = st.file_uploader(
        "Click to upload or drag and drop",
        type=file_types,
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success(f"✅ File uploaded successfully! - {uploaded_file.name}")
        
        # Preview
        if st.session_state.file_type == "Image":
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Preview", use_container_width=True)
            except Exception as e:
                st.error(f"Error: Invalid file type")
        else:
            st.video(uploaded_file)
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        model = st.selectbox(
            "Detection Model",
            ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST"),
            index=2 if st.session_state.model == 'EfficientNetAutoAttB4' else 0,
            key="model_select"
        )
        st.session_state.model = model
        
        dataset = st.radio(
            "Training Dataset",
            ("DFDC", "FFPP"),
            index=0 if st.session_state.dataset == 'DFDC' else 1,
            key="dataset_radio",
            horizontal=True
        )
        st.session_state.dataset = dataset
        
        threshold = st.slider(
            "Detection Threshold",
            0.0, 1.0, st.session_state.threshold,
            key="threshold_slider",
            step=0.01
        )
        st.session_state.threshold = threshold
        
        if st.session_state.file_type == "Video":
            frames = st.slider(
                "Frames to Analyze",
                0, 100, st.session_state.frames,
                key="frames_slider"
            )
            st.session_state.frames = frames
    
    # Check Button
    check_disabled = st.session_state.uploaded_file is None
    if st.button("🔍 Check for Deepfake", disabled=check_disabled, use_container_width=True, type="primary", key="check_btn"):
        st.session_state.page = 'loading'
        st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# Loading Page
elif st.session_state.page == 'loading':
    st.markdown("""
    <div class="gradient-bg" style="min-height: 100vh; display: flex; align-items: center; justify-content: center; flex-direction: column; padding: 2rem;">
        <div class="spinner" style="margin: 0 auto 1.5rem;"></div>
        <h2 style="font-size: 1.875rem; font-weight: bold; color: white; margin-bottom: 1rem;">Analyzing Your Media...</h2>
        <p style="font-size: 1.25rem; color: rgba(255,255,255,0.9);">Extracting features and analyzing frames</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Process the file
    if st.session_state.uploaded_file is not None:
        with st.spinner("Processing..."):
            try:
                if st.session_state.file_type == "Image":
                    result, pred = process_image(
                        image=st.session_state.uploaded_file,
                        model=st.session_state.model,
                        dataset=st.session_state.dataset,
                        threshold=st.session_state.threshold
                    )
                else:
                    os.makedirs("uploads", exist_ok=True)
                    with open(f"uploads/{st.session_state.uploaded_file.name}", "wb") as f:
                        f.write(st.session_state.uploaded_file.read())
                    
                    video_path = f"uploads/{st.session_state.uploaded_file.name}"
                    
                    result, pred = process_video(
                        video_path=video_path,
                        model=st.session_state.model,
                        dataset=st.session_state.dataset,
                        threshold=st.session_state.threshold,
                        frames=st.session_state.frames
                    )
                
                st.session_state.result = result
                st.session_state.pred = pred
                st.session_state.page = 'result'
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.session_state.page = 'main'
                st.rerun()

# Results Page
elif st.session_state.page == 'result':
    st.markdown("""
    <div class="gradient-bg" style="min-height: 100vh; padding: 3rem 1.5rem;">
        <div style="max-width: 64rem; margin: 0 auto;">
            <div class="bg-white rounded-2xl card-shadow fade-in" style="padding: 2.5rem;">
                <h1 style="font-size: 2.5rem; font-weight: bold; color: #111827; margin-bottom: 2rem; text-align: center;">Detection Results</h1>
    """, unsafe_allow_html=True)
    
    if st.session_state.result is not None and st.session_state.pred is not None:
        is_real = st.session_state.result == 'real'
        confidence = (1 - st.session_state.pred) * 100 if is_real else st.session_state.pred * 100
        percentage = round(confidence)
        
        result_class = 'real' if is_real else 'fake'
        icon_class = 'fas fa-check-circle' if is_real else 'fas fa-exclamation-triangle'
        title = 'Authentic Media' if is_real else 'Deepfake Detected'
        subtitle = 'No manipulation detected' if is_real else 'This media appears to be manipulated'
        color = '#16a34a' if is_real else '#dc2626'
        bg_color = '#f0fdf4' if is_real else '#fef2f2'
        border_color = '#16a34a' if is_real else '#dc2626'
        description = '<strong>Good news!</strong> Our analysis indicates this media is likely authentic.' if is_real else '<strong>Warning:</strong> Our AI model has detected signs of deepfake manipulation.'
        
        circumference = 2 * 3.14159 * 90
        offset = circumference - (confidence / 100 * circumference)
        
        st.markdown(f"""
        <div style="margin-bottom: 2rem; padding: 2rem; border-radius: 0.75rem; border: 4px solid {border_color}; background: {bg_color};">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <i class="{icon_class}" style="font-size: 3.75rem; color: {color};"></i>
                    <div>
                        <h2 style="font-size: 1.875rem; font-weight: bold; color: {color};">{title}</h2>
                        <p style="font-size: 1.125rem; color: #6b7280;">{subtitle}</p>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 3rem; font-weight: bold; color: {color};">{percentage}%</div>
                    <div style="color: #6b7280;">Confidence</div>
                </div>
            </div>
            <div style="display: flex; justify-content: center; margin-bottom: 1.5rem;">
                <svg width="200" height="200" class="progress-ring">
                    <circle cx="100" cy="100" r="90" stroke="#e5e7eb" stroke-width="12" fill="none"></circle>
                    <circle id="progressCircle" cx="100" cy="100" r="90" stroke="{color}" stroke-width="12" fill="none" 
                            stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" 
                            class="progress-ring-circle" stroke-linecap="round"></circle>
                    <text x="100" y="100" text-anchor="middle" dy=".3em" style="font-size: 2.25rem; font-weight: bold;" fill="#1e293b">{percentage}%</text>
                </svg>
            </div>
            <div style="text-align: center; color: #374151; font-size: 1.125rem;">
                {description}
            </div>
        </div>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        """, unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Analyze Another File", use_container_width=True):
                st.session_state.uploaded_file = None
                st.session_state.result = None
                st.session_state.pred = None
                st.session_state.page = 'main'
                st.rerun()
        with col2:
            if st.button("Download Report", use_container_width=True, type="primary"):
                st.info("Report download functionality would be implemented here")
    
    st.markdown('</div></div></div>', unsafe_allow_html=True)
