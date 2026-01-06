import streamlit as st
from PIL import Image
from api import process_image, process_video
import time

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, professional UI
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Hero Section Styles */
    .hero-container {
        text-align: center;
        padding: 80px 20px;
        color: white;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.8rem;
        font-weight: 300;
        margin-bottom: 30px;
        opacity: 0.95;
    }
    
    .hero-description {
        font-size: 1.2rem;
        margin-bottom: 50px;
        opacity: 0.9;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    .start-button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 18px 50px;
        font-size: 1.3rem;
        font-weight: 600;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .start-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    }
    
    /* Main Container Styles */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 40px;
        margin: 20px auto;
        max-width: 1200px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Upload Section Styles */
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: #f8f9ff;
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: #f0f2ff;
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 20px;
    }
    
    .success-message {
        background: #10b981;
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin-top: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Advanced Settings Styles */
    .advanced-settings {
        background: #f8f9ff;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .info-tooltip {
        background: #e0e7ff;
        color: #4338ca;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9rem;
        border-left: 4px solid #4338ca;
    }
    
    /* Result Card Styles */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 40px;
        margin: 30px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .result-card.real {
        border: 4px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
    }
    
    .result-card.fake {
        border: 4px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    
    .result-icon {
        font-size: 5rem;
        margin-bottom: 20px;
    }
    
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .result-percentage {
        font-size: 3rem;
        font-weight: 800;
        margin: 20px 0;
    }
    
    /* Progress Bar Styles */
    .progress-container {
        margin: 30px 0;
        padding: 20px;
        background: #f8f9ff;
        border-radius: 15px;
    }
    
    /* How It Works Styles */
    .how-it-works {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    
    .how-it-works-button {
        background: rgba(255, 255, 255, 0.9);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .how-it-works-button:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .how-it-works-content {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin-top: 10px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        max-width: 350px;
    }
    
    .step-item {
        padding: 15px;
        margin: 10px 0;
        background: #f8f9ff;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Loading Spinner */
    .spinner-container {
        text-align: center;
        padding: 40px;
    }
    
    .spinner {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #667eea;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 0 auto 20px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Button Styles */
    .check-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-weight: 600;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
    }
    
    .check-button:hover:not(:disabled) {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .check-button:disabled {
        background: #cbd5e1;
        cursor: not-allowed;
        box-shadow: none;
    }
    
    /* File Type Selection */
    .file-type-container {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin-bottom: 30px;
    }
    
    .file-type-card {
        padding: 20px 40px;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }
    
    .file-type-card.selected {
        border-color: #667eea;
        background: #f0f2ff;
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
if 'show_how_it_works' not in st.session_state:
    st.session_state.show_how_it_works = False

# How It Works Section (only show on main and result pages)
if st.session_state.page != 'hero':
    # Position How It Works button in top right
    st.markdown("""
    <div style="position: fixed; top: 20px; right: 20px; z-index: 1000;">
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("❓ How It Works", key="how_it_works_btn", use_container_width=False):
            st.session_state.show_how_it_works = not st.session_state.show_how_it_works
    
    if st.session_state.show_how_it_works:
        st.info("""
        **How It Works:**
        
        1. **Upload** - Select an image or video file to analyze
        
        2. **Analyze** - Our AI extracts frames and features from your media
        
        3. **Predict** - Advanced models determine if content is real or deepfake
        """)

# Hero Page
if st.session_state.page == 'hero':
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔍 Deepfake Detector</div>
        <div class="hero-subtitle">Detect AI-generated images and videos</div>
        <div class="hero-description">
            Upload an image or video to check if it has been manipulated using deepfake techniques. 
            Our advanced AI models analyze your media to provide accurate detection results.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start", key="start_button", use_container_width=True):
            st.session_state.page = 'main'
            st.rerun()

# Main Page
elif st.session_state.page == 'main':
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown("""
    <h1 style="text-align: center; color: #667eea; margin-bottom: 40px;">
        🔍 Deepfake Detector
    </h1>
    """, unsafe_allow_html=True)
    
    # File Type Selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        file_type = st.radio(
            "Select file type:",
            ("Image", "Video"),
            horizontal=True,
            key="file_type_radio",
            index=0 if st.session_state.file_type == 'Image' else 1,
            label_visibility="collapsed"
        )
        # Clear uploaded file if file type changed
        if file_type != st.session_state.file_type:
            st.session_state.uploaded_file = None
            st.session_state.file_uploaded = False
        st.session_state.file_type = file_type
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    # File Upload with Icons
    if file_type == "Image":
        upload_icon = "📷"
        upload_text = "Upload Image"
        accepted_types = ["jpg", "jpeg", "png"]
    else:
        upload_icon = "🎬"
        upload_text = "Upload Video"
        accepted_types = ["mp4"]
    
    st.markdown(f"""
    <div style="text-align: center;">
        <div class="upload-icon">{upload_icon}</div>
        <h3 style="color: #667eea; margin-bottom: 10px;">{upload_text}</h3>
        <p style="color: #64748b;">Supported formats: {', '.join(accepted_types).upper()}</p>
    </div>
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
            ✅ File uploaded successfully!
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
    
    # Advanced Settings (Expandable)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown('<div class="advanced-settings">', unsafe_allow_html=True)
        
        # Model Selection with Tooltip
        model_options = ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
        model_index = model_options.index(st.session_state.model) if st.session_state.model in model_options else 2
        model = st.selectbox(
            "Select Model",
            model_options,
            index=model_index,
            key="model_select"
        )
        st.session_state.model = model
        st.markdown("""
        <div class="info-tooltip">
            <strong>💡 Model:</strong> Choose the AI model used for detection. 
            EfficientNetAutoAttB4 is recommended for best accuracy.
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset Selection with Tooltip
        dataset = st.radio(
            "Select Dataset",
            ("DFDC", "FFPP"),
            index=0 if st.session_state.dataset == 'DFDC' else 1,
            key="dataset_radio",
            horizontal=True
        )
        st.session_state.dataset = dataset
        st.markdown("""
        <div class="info-tooltip">
            <strong>💡 Dataset:</strong> Select the training dataset. 
            DFDC (Deepfake Detection Challenge) is recommended for general use.
        </div>
        """, unsafe_allow_html=True)
        
        # Threshold Slider with Tooltip
        threshold = st.slider(
            "Select Threshold",
            0.0, 1.0, st.session_state.threshold,
            key="threshold_slider",
            step=0.01
        )
        st.session_state.threshold = threshold
        st.markdown(f"""
        <div class="info-tooltip">
            <strong>💡 Threshold:</strong> Set the sensitivity level (Current: {threshold:.2f}). 
            Lower values are more sensitive to deepfakes, higher values require stronger evidence.
        </div>
        """, unsafe_allow_html=True)
        
        # Video Frames (only for video)
        if file_type == "Video":
            frames = st.slider(
                "Select Frames",
                0, 100, st.session_state.frames,
                key="frames_slider"
            )
            st.session_state.frames = frames
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check for Deepfake Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        check_disabled = not st.session_state.file_uploaded
        
        if st.button(
            "🔍 Check for Deepfake",
            disabled=check_disabled,
            use_container_width=True,
            key="check_button"
        ):
            st.session_state.show_result = True
            st.session_state.page = 'result'
            st.rerun()
    
    if check_disabled:
        st.info("👆 Please upload a file to continue")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Result Page
elif st.session_state.page == 'result':
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Process the file if not already processed
    if st.session_state.uploaded_file is not None and st.session_state.result is None:
        # Show loading spinner
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div class="spinner-container">
            <div class="spinner"></div>
            <h3 style="color: #667eea;">Analyzing your file...</h3>
            <p style="color: #64748b;">This may take a few moments. Please wait.</p>
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
            if st.button("🔙 Go Back", key="error_back"):
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
        result_icon = '✅' if is_real else '⚠️'
        result_color = '#10b981' if is_real else '#ef4444'
        result_text = 'REAL' if is_real else 'DEEPFAKE'
        
        # Convert probability to percentage
        deepfake_percentage = st.session_state.pred * 100
        real_percentage = (1 - st.session_state.pred) * 100
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <div class="result-icon">{result_icon}</div>
            <div class="result-text" style="color: {result_color};">{result_text}</div>
            <div class="result-percentage" style="color: {result_color};">
                {real_percentage:.1f}% Real / {deepfake_percentage:.1f}% Deepfake
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar Visualization
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Confidence Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h4 style="color: #10b981;">Real Content</h4>
                <div style="background: #e5e7eb; border-radius: 10px; height: 30px; margin: 10px 0;">
                    <div style="background: #10b981; height: 30px; width: {real_percentage}%; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {real_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h4 style="color: #ef4444;">Deepfake Content</h4>
                <div style="background: #e5e7eb; border-radius: 10px; height: 30px; margin: 10px 0;">
                    <div style="background: #ef4444; height: 30px; width: {deepfake_percentage}%; border-radius: 10px; 
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                        {deepfake_percentage:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Analyze Another File", use_container_width=True, key="analyze_another"):
                st.session_state.page = 'main'
                st.session_state.file_uploaded = False
                st.session_state.uploaded_file = None
                st.session_state.show_result = False
                st.session_state.result = None
                st.session_state.pred = None
                st.rerun()
            
            if st.button("🏠 Back to Home", use_container_width=True, key="back_home"):
                st.session_state.page = 'hero'
                st.session_state.file_uploaded = False
                st.session_state.uploaded_file = None
                st.session_state.show_result = False
                st.session_state.result = None
                st.session_state.pred = None
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
