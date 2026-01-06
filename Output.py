import streamlit as st
from PIL import Image
from api import process_image, process_video
import time

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Exact CSS from HTML design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    body, .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0;
        padding: 0;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* Hero Page */
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1.1;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .card-shadow {
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: all 0.3s ease;
    }
    
    .btn-danger {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        transition: all 0.3s ease;
        color: white;
        font-size: 1.25rem;
        font-weight: bold;
        padding: 1.25rem 4rem;
        border-radius: 0.75rem;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.4);
        border: none;
    }
    
    .btn-danger:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(220, 38, 38, 0.5);
    }
    
    .upload-zone {
        border: 3px dashed #cbd5e1;
        transition: all 0.3s ease;
        border-radius: 0.75rem;
        padding: 3rem;
        text-align: center;
        background: white;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    .file-type-btn {
        flex: 1;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        border: 3px solid #e5e7eb;
        background: white;
        font-weight: 600;
        font-size: 1.125rem;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .file-type-btn.active {
        border-color: #667eea;
        background: #f3f4ff;
        color: #667eea;
    }
    
    .file-type-btn:hover {
        border-color: #667eea;
        background: #f8f9ff;
    }
    
    .spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto 1.5rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 0.75rem;
        border: 4px solid;
        margin-bottom: 2rem;
    }
    
    .result-card.real {
        border-color: #16a34a;
        background: #f0fdf4;
    }
    
    .result-card.fake {
        border-color: #dc2626;
        background: #fef2f2;
    }
    
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    .how-it-works-panel {
        position: fixed;
        top: 0;
        right: -400px;
        width: 400px;
        height: 100vh;
        background: white;
        box-shadow: -4px 0 20px rgba(0,0,0,0.2);
        transition: right 0.3s ease;
        z-index: 1000;
        overflow-y: auto;
        padding: 2rem;
    }
    
    .how-it-works-panel.open {
        right: 0;
    }
    
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 999;
        display: none;
    }
    
    .overlay.active {
        display: block;
    }
    
    .step-item {
        display: flex;
        align-items: start;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .step-number {
        flex-shrink: 0;
        width: 3rem;
        height: 3rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.25rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        font-size: 1.25rem;
        font-weight: bold;
        padding: 1.25rem;
        border-radius: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    /* Large text */
    h1 { font-size: 2.5rem !important; font-weight: 800 !important; }
    h2 { font-size: 2rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.5rem !important; font-weight: 600 !important; }
    
    .main-container {
        background: white;
        border-radius: 1.25rem;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        margin: 1.5rem auto;
        max-width: 80rem;
    }
    
    @media (max-width: 768px) {
        .hero-title { font-size: 3rem; }
        .how-it-works-panel { width: 100%; right: -100%; }
    }
</style>

<script>
function openHowItWorks() {
    document.getElementById('howItWorksPanel').classList.add('open');
    document.getElementById('overlay').classList.add('active');
}

function closeHowItWorks() {
    document.getElementById('howItWorksPanel').classList.remove('open');
    document.getElementById('overlay').classList.remove('active');
}

window.addEventListener('click', function(event) {
    const panel = document.getElementById('howItWorksPanel');
    const overlay = document.getElementById('overlay');
    if (event.target === overlay) {
        closeHowItWorks();
    }
});
</script>
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
if 'show_how_it_works' not in st.session_state:
    st.session_state.show_how_it_works = False

# Hero Page
if st.session_state.page == 'hero':
    st.markdown("""
    <div style="min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem;">
        <div style="text-align: center; max-width: 64rem; margin: 0 auto;">
            <h1 class="hero-title">Deepfake Detector</h1>
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
    
    # Apply red button styling
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
    }
    </style>
    """, unsafe_allow_html=True)

# Main Page
elif st.session_state.page == 'main':
    # Header
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown('<h1 style="color: white; font-size: 2.5rem; font-weight: 800;">Deepfake Detector</h1>', unsafe_allow_html=True)
    with header_col2:
        if st.button("❓ How It Works", key="how_it_works_btn"):
            st.session_state.show_how_it_works = True
            st.rerun()
    
    # How It Works Panel
    if st.session_state.show_how_it_works:
        st.markdown("""
        <div class="overlay active" id="overlay"></div>
        <div class="how-it-works-panel open" id="howItWorksPanel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.5rem; font-weight: bold; color: #111827;">How It Works</h2>
                <button onclick="closeHowItWorks()" style="background: none; border: none; font-size: 1.5rem; color: #6b7280; cursor: pointer;">×</button>
            </div>
            <div class="step-item">
                <div class="step-number">1</div>
                <div>
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Upload Your Media</h3>
                    <p style="color: #4b5563;">Select an image or video file from your device. We support JPG, PNG, and MP4 formats.</p>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <div>
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Analyze Frames & Features</h3>
                    <p style="color: #4b5563;">Our AI model examines facial features, artifacts, and inconsistencies that indicate manipulation.</p>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <div>
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Get Instant Results</h3>
                    <p style="color: #4b5563;">Receive a detailed analysis with confidence scores showing whether the media is authentic or manipulated.</p>
                </div>
            </div>
            <div style="margin-top: 2rem; padding: 1rem; background: #eff6ff; border-radius: 0.5rem;">
                <h4 style="font-weight: 600; color: #111827; margin-bottom: 0.5rem;">Advanced Settings</h4>
                <p style="font-size: 0.875rem; color: #4b5563;">Fine-tune detection with custom models, datasets, and threshold values.</p>
            </div>
            <button onclick="closeHowItWorks()" style="margin-top: 1.5rem; width: 100%; padding: 0.75rem; background: #667eea; color: white; border: none; border-radius: 0.5rem; font-weight: 600; cursor: pointer;">Close</button>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Close", key="close_how_it_works"):
            st.session_state.show_how_it_works = False
            st.rerun()
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # File Type Selection
    st.markdown('<h2 style="font-size: 1.5rem; font-weight: bold; color: #111827; margin-bottom: 1rem;">Select File Type</h2>', unsafe_allow_html=True)
    
    file_type_col1, file_type_col2 = st.columns(2)
    
    # Apply custom styling to buttons
    image_style = f"""
    <style>
    button[key="image_btn"] {{
        border: 3px solid {'#667eea' if st.session_state.file_type == 'Image' else '#e5e7eb'} !important;
        background: {'#f3f4ff' if st.session_state.file_type == 'Image' else 'white'} !important;
        color: {'#667eea' if st.session_state.file_type == 'Image' else '#374151'} !important;
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
    }}
    button[key="video_btn"] {{
        border: 3px solid {'#667eea' if st.session_state.file_type == 'Video' else '#e5e7eb'} !important;
        background: {'#f3f4ff' if st.session_state.file_type == 'Video' else 'white'} !important;
        color: {'#667eea' if st.session_state.file_type == 'Video' else '#374151'} !important;
        font-size: 1.125rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
    }}
    </style>
    """
    st.markdown(image_style, unsafe_allow_html=True)
    
    with file_type_col1:
        if st.button("📷 Image", key="image_btn", use_container_width=True):
            st.session_state.file_type = 'Image'
            st.session_state.uploaded_file = None
            st.rerun()
    with file_type_col2:
        if st.button("🎥 Video", key="video_btn", use_container_width=True):
            st.session_state.file_type = 'Video'
            st.session_state.uploaded_file = None
            st.rerun()
    
    # File Upload
    st.markdown('<h2 style="font-size: 1.5rem; font-weight: bold; color: #111827; margin-bottom: 1rem; margin-top: 2rem;">Upload File</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-zone" style="text-align: center; padding: 3rem; border: 3px dashed #cbd5e1; border-radius: 0.75rem; background: white; cursor: pointer;">
        <div style="font-size: 3.75rem; color: #9ca3af; margin-bottom: 1rem;">☁️</div>
        <p style="font-size: 1.25rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Click to upload or drag and drop</p>
        <p style="color: #6b7280;">JPG, PNG or MP4 (Max 50MB)</p>
    </div>
    """, unsafe_allow_html=True)
    
    file_types = ["jpg", "jpeg", "png"] if st.session_state.file_type == 'Image' else ["mp4"]
    uploaded_file = st.file_uploader(
        "Choose a file",
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# Loading Page
elif st.session_state.page == 'loading':
    st.markdown("""
    <div style="min-height: 100vh; display: flex; align-items: center; justify-content: center; flex-direction: column;">
        <div class="spinner"></div>
        <h2 style="font-size: 1.875rem; font-weight: bold; color: white; margin-bottom: 1rem;">Analyzing Your Media...</h2>
        <p style="font-size: 1.25rem; color: rgba(255,255,255,0.9);" id="loadingMessage">Extracting features and analyzing frames</p>
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
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: bold; color: #111827; margin-bottom: 2rem; text-align: center;">Detection Results</h1>', unsafe_allow_html=True)
    
    if st.session_state.result is not None and st.session_state.pred is not None:
        is_real = st.session_state.result == 'real'
        confidence = (1 - st.session_state.pred) * 100 if is_real else st.session_state.pred * 100
        percentage = round(confidence)
        
        result_class = 'real' if is_real else 'fake'
        icon = '✅' if is_real else '⚠️'
        title = 'Authentic Media' if is_real else 'Deepfake Detected'
        subtitle = 'No manipulation detected' if is_real else 'This media appears to be manipulated'
        color = '#16a34a' if is_real else '#dc2626'
        bg_color = '#f0fdf4' if is_real else '#fef2f2'
        border_color = '#16a34a' if is_real else '#dc2626'
        
        st.markdown(f"""
        <div class="result-card {result_class}" style="border-color: {border_color}; background: {bg_color};">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 3.75rem;">{icon}</div>
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
            <div style="text-align: center; color: #374151; font-size: 1.125rem;">
                {'<strong>Good news!</strong> Our analysis indicates this media is likely authentic. No significant signs of deepfake manipulation were detected.' if is_real else '<strong>Warning:</strong> Our AI model has detected signs of deepfake manipulation in this media. The content may have been artificially generated or altered using machine learning techniques.'}
            </div>
        </div>
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
    
    st.markdown('</div>', unsafe_allow_html=True)
