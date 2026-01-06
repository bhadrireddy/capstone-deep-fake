import streamlit as st
from PIL import Image
from api import process_image, process_video

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS - Clean, Modern Design
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
        background: #f5f7fa;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        min-height: 100vh;
    }
    
    /* Header Styles */
    .app-header {
        background: white;
        border-bottom: 1px solid #e2e8f0;
        padding: 24px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .logo-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        font-weight: 700;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    .header-link {
        color: #0891b2;
        text-decoration: none;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        padding: 8px 16px;
        border-radius: 8px;
        transition: background 0.2s;
    }
    
    .header-link:hover {
        background: #f0f9ff;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 120px 40px 80px;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 56px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 24px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 24px;
        font-weight: 400;
        color: #64748b;
        margin-bottom: 32px;
    }
    
    .hero-description {
        font-size: 18px;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 48px;
    }
    
    /* Main Container */
    .main-container {
        background: white;
        border-radius: 16px;
        padding: 48px;
        margin: 40px auto;
        max-width: 1000px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .page-title {
        font-size: 36px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 40px;
        text-align: center;
    }
    
    /* File Type Selection */
    .file-type-container {
        display: flex;
        gap: 16px;
        justify-content: center;
        margin-bottom: 32px;
    }
    
    /* Upload Section */
    .upload-section {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 60px 40px;
        text-align: center;
        background: #f8fafc;
        margin-bottom: 32px;
        transition: all 0.2s;
    }
    
    .upload-section:hover {
        border-color: #0891b2;
        background: #f0f9ff;
    }
    
    .upload-icon-wrapper {
        width: 80px;
        height: 80px;
        margin: 0 auto 24px;
        background: #e0f2fe;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .upload-icon {
        width: 48px;
        height: 48px;
        color: #0891b2;
    }
    
    .upload-title {
        font-size: 24px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .upload-subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 24px;
    }
    
    .success-message {
        background: #d1fae5;
        color: #065f46;
        padding: 16px 24px;
        border-radius: 8px;
        margin-top: 24px;
        font-weight: 500;
        font-size: 16px;
        display: inline-block;
        border: 1px solid #a7f3d0;
    }
    
    /* Buttons */
    .primary-button {
        background: #0891b2;
        color: white;
        padding: 16px 48px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
        font-family: 'Inter', sans-serif;
    }
    
    .primary-button:hover:not(:disabled) {
        background: #0e7490;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(8, 145, 178, 0.3);
    }
    
    .primary-button:disabled {
        background: #cbd5e1;
        cursor: not-allowed;
        color: #94a3b8;
    }
    
    .secondary-button {
        background: white;
        color: #0891b2;
        padding: 16px 48px;
        font-size: 18px;
        font-weight: 600;
        border: 2px solid #0891b2;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
        font-family: 'Inter', sans-serif;
    }
    
    .secondary-button:hover {
        background: #f0f9ff;
    }
    
    /* Advanced Settings */
    .advanced-settings-content {
        background: #f8fafc;
        border-radius: 12px;
        padding: 32px;
        margin-top: 16px;
    }
    
    .info-text {
        background: #e0f2fe;
        color: #0c4a6e;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 12px;
        font-size: 14px;
        line-height: 1.6;
        border-left: 4px solid #0891b2;
    }
    
    /* Result Card */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 48px;
        margin: 32px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 2px solid;
        text-align: center;
    }
    
    .result-card.real {
        border-color: #10b981;
        background: linear-gradient(to bottom, #ecfdf5 0%, #ffffff 100%);
    }
    
    .result-card.fake {
        border-color: #ef4444;
        background: linear-gradient(to bottom, #fef2f2 0%, #ffffff 100%);
    }
    
    .result-status {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 16px;
    }
    
    .result-label {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 24px;
    }
    
    .result-percentage {
        font-size: 40px;
        font-weight: 700;
        margin: 24px 0;
    }
    
    /* Progress Bars */
    .progress-container {
        margin: 32px 0;
        padding: 32px;
        background: #f8fafc;
        border-radius: 12px;
    }
    
    .progress-title {
        font-size: 20px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 24px;
        text-align: center;
    }
    
    .progress-item {
        margin-bottom: 24px;
    }
    
    .progress-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #475569;
    }
    
    .progress-bar-wrapper {
        background: #e2e8f0;
        border-radius: 8px;
        height: 40px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 16px;
        transition: width 0.3s ease;
    }
    
    /* Loading Spinner */
    .spinner-container {
        text-align: center;
        padding: 80px 40px;
    }
    
    .spinner {
        border: 4px solid #e2e8f0;
        border-top: 4px solid #0891b2;
        border-radius: 50%;
        width: 64px;
        height: 64px;
        animation: spin 1s linear infinite;
        margin: 0 auto 24px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 20px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .loading-subtext {
        font-size: 16px;
        color: #64748b;
    }
    
    /* Modal Styles */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(4px);
    }
    
    .modal-content {
        background: white;
        border-radius: 16px;
        padding: 40px;
        max-width: 500px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 25px rgba(0,0,0,0.15);
        position: relative;
        animation: modalFadeIn 0.2s ease;
    }
    
    @keyframes modalFadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
    
    .modal-title {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    .modal-close-btn {
        background: #f1f5f9;
        border: none;
        font-size: 24px;
        color: #64748b;
        cursor: pointer;
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s;
        font-weight: 600;
        line-height: 1;
    }
    
    .modal-close-btn:hover {
        background: #e2e8f0;
        color: #1e293b;
    }
    
    .step-item {
        padding: 20px;
        margin: 16px 0;
        background: #f8fafc;
        border-radius: 12px;
        border-left: 4px solid #0891b2;
    }
    
    .step-number {
        font-size: 18px;
        font-weight: 700;
        color: #0891b2;
        margin-bottom: 8px;
    }
    
    .step-description {
        font-size: 16px;
        color: #475569;
        line-height: 1.6;
    }
    
    /* Streamlit Component Overrides */
    .stRadio > div {
        flex-direction: row;
        gap: 16px;
    }
    
    .stRadio > div > label {
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        background: white;
        transition: all 0.2s;
    }
    
    .stRadio > div > label:hover {
        border-color: #0891b2;
        background: #f0f9ff;
    }
    
    .stRadio > div > label[data-baseweb="radio"] {
        font-size: 18px;
    }
    
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin: 24px 0;
    }
    
    .stExpander > summary {
        font-size: 18px;
        font-weight: 600;
        color: #1e293b;
        padding: 16px 20px;
    }
    
    .stSelectbox label,
    .stSlider label {
        font-size: 16px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .stFileUploader {
        margin-top: 24px;
    }
</style>

""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'hero'
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = 'Image'
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'result' not in st.session_state:
    st.session_state.result = None
if 'pred' not in st.session_state:
    st.session_state.pred = None
if 'model' not in st.session_state:
    st.session_state.model = 'EfficientNetAutoAttB4'
if 'dataset' not in st.session_state:
    st.session_state.dataset = 'DFDC'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'frames' not in st.session_state:
    st.session_state.frames = 50
if 'show_modal' not in st.session_state:
    st.session_state.show_modal = False

# Header Component (shown on main and result pages)
def render_header():
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("""
        <div class="header-left">
            <div class="logo-icon">DF</div>
            <h1 class="header-title">Deepfake Detector</h1>
        </div>
        """, unsafe_allow_html=True)
    with header_col2:
        st.markdown('<div style="text-align: right; padding-top: 12px;">', unsafe_allow_html=True)
        if st.button("How It Works", key="how_it_works_btn", use_container_width=True):
            st.session_state.show_modal = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Modal Component
def render_modal():
    if st.session_state.show_modal:
        # Create modal-like container
        st.markdown("""
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                    background: rgba(0, 0, 0, 0.6); z-index: 9999; 
                    display: flex; align-items: center; justify-content: center;">
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 40px; 
                        box-shadow: 0 20px 25px rgba(0,0,0,0.25); 
                        position: relative; z-index: 10000; margin-top: -200px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px;">
                    <h2 style="font-size: 28px; font-weight: 700; color: #1e293b; margin: 0;">How It Works</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: white; border-radius: 16px; padding: 0 40px 40px; 
                        box-shadow: 0 20px 25px rgba(0,0,0,0.25); 
                        position: relative; z-index: 10000; margin-top: -16px;">
                <div class="step-item">
                    <div class="step-number">1. Upload</div>
                    <div class="step-description">Select an image or video file from your device to analyze for deepfake manipulation.</div>
                </div>
                <div class="step-item">
                    <div class="step-number">2. Analyze</div>
                    <div class="step-description">Our AI system extracts frames and features from your media, examining patterns and anomalies that indicate manipulation.</div>
                </div>
                <div class="step-item">
                    <div class="step-number">3. Predict</div>
                    <div class="step-description">Advanced deep learning models analyze the extracted features to determine whether the content is authentic or has been manipulated using deepfake techniques.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Close", key="close_modal", use_container_width=True, type="primary"):
                st.session_state.show_modal = False
                st.rerun()

# Hero Page
if st.session_state.page == 'hero':
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Deepfake Detector</h1>
        <h2 class="hero-subtitle">Detect AI-generated images and videos</h2>
        <p class="hero-description">
            Upload an image or video to check if it has been manipulated using deepfake techniques. 
            Our advanced AI models analyze your media to provide accurate detection results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start", key="start_button", use_container_width=True, type="primary"):
            st.session_state.page = 'main'
            st.rerun()

# Main Page
elif st.session_state.page == 'main':
    render_header()
    render_modal()
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="page-title">Deepfake Detector</h1>', unsafe_allow_html=True)
    
    # File Type Selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        file_type = st.radio(
            "Select file type:",
            ("Image", "Video"),
            horizontal=True,
            key="file_type_radio",
            index=0 if st.session_state.file_type == 'Image' else 1,
            label_visibility="visible"
        )
        # Clear uploaded file if file type changed
        if file_type != st.session_state.file_type:
            st.session_state.uploaded_file = None
            st.session_state.file_uploaded = False
        st.session_state.file_type = file_type
    
    # Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    if file_type == "Image":
        upload_text = "Upload Image"
        accepted_types = ["jpg", "jpeg", "png"]
        icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>'
    else:
        upload_text = "Upload Video"
        accepted_types = ["mp4"]
        icon_svg = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>'
    
    st.markdown(f"""
    <div class="upload-icon-wrapper">
        <div class="upload-icon">{icon_svg}</div>
    </div>
    <h3 class="upload-title">{upload_text}</h3>
    <p class="upload-subtitle">Supported formats: {', '.join(accepted_types).upper()} • Max size: 100MB</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        f"Choose a {file_type.lower()}...",
        type=accepted_types,
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.file_uploaded = True
        
        # Show success message
        st.markdown("""
        <div class="success-message">
            File uploaded successfully
        </div>
        """, unsafe_allow_html=True)
        
        # Display preview
        if file_type == "Image":
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error: Invalid file type")
        else:
            st.video(uploaded_file)
    else:
        st.session_state.file_uploaded = False
        st.session_state.show_result = False
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Settings
    with st.expander("Advanced Settings", expanded=False):
        st.markdown('<div class="advanced-settings-content">', unsafe_allow_html=True)
        
        # Model Selection
        model_options = ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
        model_index = model_options.index(st.session_state.model) if st.session_state.model in model_options else 2
        model = st.selectbox(
            "Model",
            model_options,
            index=model_index,
            key="model_select"
        )
        st.session_state.model = model
        st.markdown("""
        <div class="info-text">
            <strong>Model:</strong> Choose the AI model used for detection. EfficientNetAutoAttB4 is recommended for best accuracy.
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset Selection
        dataset = st.radio(
            "Dataset",
            ("DFDC", "FFPP"),
            index=0 if st.session_state.dataset == 'DFDC' else 1,
            key="dataset_radio",
            horizontal=True
        )
        st.session_state.dataset = dataset
        st.markdown("""
        <div class="info-text">
            <strong>Dataset:</strong> Select the training dataset. DFDC (Deepfake Detection Challenge) is recommended for general use.
        </div>
        """, unsafe_allow_html=True)
        
        # Threshold Slider
        threshold = st.slider(
            "Threshold",
            0.0, 1.0, st.session_state.threshold,
            key="threshold_slider",
            step=0.01
        )
        st.session_state.threshold = threshold
        st.markdown(f"""
        <div class="info-text">
            <strong>Threshold:</strong> Set the sensitivity level (Current: {threshold:.2f}). Lower values are more sensitive to deepfakes, higher values require stronger evidence.
        </div>
        """, unsafe_allow_html=True)
        
        # Video Frames
        if file_type == "Video":
            frames = st.slider(
                "Number of Frames",
                0, 100, st.session_state.frames,
                key="frames_slider"
            )
            st.session_state.frames = frames
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_disabled = not st.session_state.file_uploaded
        
        if st.button(
            "Analyze Media",
            disabled=check_disabled,
            use_container_width=True,
            key="check_button",
            type="primary"
        ):
            st.session_state.show_result = True
            st.session_state.page = 'result'
            st.rerun()
    
    if check_disabled:
        st.info("Please upload a file to continue")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Result Page
elif st.session_state.page == 'result':
    render_header()
    render_modal()
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Process the file if not already processed
    if st.session_state.uploaded_file is not None and st.session_state.result is None:
        # Show loading spinner
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="loading-text">Analyzing your file</div>
            <div class="loading-subtext">This may take a few moments. Please wait.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process the file
        try:
            if st.session_state.file_type == "Image":
                result, pred = process_image(
                    image=st.session_state.uploaded_file,
                    model=st.session_state.model,
                    dataset=st.session_state.dataset,
                    threshold=st.session_state.threshold
                )
            else:
                # Save video file
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
            loading_placeholder.empty()
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Error processing file: {str(e)}")
            if st.button("Go Back", key="error_back", type="secondary"):
                st.session_state.page = 'main'
                st.session_state.result = None
                st.session_state.pred = None
                st.rerun()
            st.stop()
    
    # Show results only if available
    if st.session_state.result is not None and st.session_state.pred is not None:
        # Result Card
        is_real = st.session_state.result == 'real'
        result_class = 'real' if is_real else 'fake'
        result_color = '#10b981' if is_real else '#ef4444'
        result_text = 'AUTHENTIC' if is_real else 'DEEPFAKE DETECTED'
        
        # Convert probability to percentage
        deepfake_percentage = st.session_state.pred * 100
        real_percentage = (1 - st.session_state.pred) * 100
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <div class="result-status" style="color: {result_color};">{result_text}</div>
            <div class="result-percentage" style="color: {result_color};">
                {real_percentage:.1f}% Real / {deepfake_percentage:.1f}% Deepfake
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar Visualization
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.markdown('<div class="progress-title">Confidence Breakdown</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="progress-item">
                <div class="progress-label">Real Content</div>
                <div class="progress-bar-wrapper">
                    <div class="progress-bar-fill" style="background: #10b981; width: {real_percentage}%;">
                        {real_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="progress-item">
                <div class="progress-label">Deepfake Content</div>
                <div class="progress-bar-wrapper">
                    <div class="progress-bar-fill" style="background: #ef4444; width: {deepfake_percentage}%;">
                        {deepfake_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Analyze Another File", use_container_width=True, key="analyze_another", type="primary"):
                st.session_state.page = 'main'
                st.session_state.file_uploaded = False
                st.session_state.uploaded_file = None
                st.session_state.show_result = False
                st.session_state.result = None
                st.session_state.pred = None
                st.rerun()
            
            if st.button("Back to Home", use_container_width=True, key="back_home", type="secondary"):
                st.session_state.page = 'hero'
                st.session_state.file_uploaded = False
                st.session_state.uploaded_file = None
                st.session_state.show_result = False
                st.session_state.result = None
                st.session_state.pred = None
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
