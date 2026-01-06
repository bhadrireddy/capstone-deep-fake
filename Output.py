import streamlit as st
from PIL import Image
from api import process_image, process_video
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector - AI-Powered Media Authentication",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Session State Management ---
# We use this to mimic the JavaScript page navigation (Hero -> Main -> Results)
if 'page' not in st.session_state:
    st.session_state.page = 'hero'
if 'result_data' not in st.session_state:
    st.session_state.result_data = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = "Image"

# Ensure uploads directory exists (from your original logic)
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# --- CSS & Styling Injection ---
# This injects your specific frontend.html styles and Tailwind CSS
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        
        /* Global Font & Background */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* The Main Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }

        /* Hide standard Streamlit Header/Footer/Menu for clean look */
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom Classes from your HTML */
        .hero-title {
            font-size: 4.5rem;
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: -0.02em;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
            color: white;
        }
        
        .card-shadow {
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            background-color: white;
            border-radius: 1rem;
            padding: 2.5rem;
        }

        /* Styling Streamlit Widgets to match frontend.html */
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.75rem;
            font-weight: 700;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Red 'Start Detection' Button Style Override */
        .btn-danger button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        }
        .btn-danger button:hover {
            box-shadow: 0 8px 20px rgba(220, 38, 38, 0.4) !important;
        }

        /* File Uploader Styling */
        [data-testid="stFileUploader"] {
            background-color: #f8f9ff;
            border: 3px dashed #cbd5e1;
            border-radius: 1rem;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #667eea;
            background-color: #f0f4ff;
        }
        
        /* Inputs/Selects */
        .stSelectbox > div > div {
            border-radius: 0.5rem;
            border: 2px solid #e5e7eb;
        }
        
        /* Radio Buttons (File Type) */
        .stRadio > div {
            flex-direction: row;
            gap: 20px;
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
    # We use columns to center the content
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding-top: 10vh; margin-bottom: 2rem;">
                <h1 class="hero-title">Deepfake Detector</h1>
                <h2 style="font-size: 2.25rem; font-weight: 600; color: white; margin-bottom: 2rem;">
                    Detect AI-generated images and videos
                </h2>
                <p style="font-size: 1.25rem; color: rgba(255,255,255,0.9); max-width: 42rem; margin: 0 auto 3rem auto; line-height: 1.6;">
                    Upload an image or video to check if it has been manipulated using deepfake techniques. 
                    Our advanced AI models analyze media files to detect artificial manipulation with high accuracy.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Center the start button
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            st.button("Start Detection ➜", on_click=go_to_main, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# --- PAGE 2 & 3: MAIN UPLOAD & RESULTS ---
elif st.session_state.page == 'main' or st.session_state.page == 'results':
    
    # Container for the white card
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Header within the main page
        st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2.25rem; font-weight: 700; color: white;">Deepfake Detector</h1>
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 0.5rem; color: white;">
                    <i class="fas fa-robot"></i> AI Powered
                </div>
            </div>
        """, unsafe_allow_html=True)

        # White Card Container
        with st.container():
            st.markdown('<div class="card-shadow">', unsafe_allow_html=True)

            # --- INPUT SECTION (Only show if no result yet) ---
            if st.session_state.result_data is None:
                st.markdown('<h2 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #111827;">Select File Type</h2>', unsafe_allow_html=True)
                
                # Custom File Type Selection
                file_type = st.radio(
                    "Select file type:", 
                    ("Image", "Video"), 
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                st.markdown('<h2 style="font-size: 1.5rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 1rem; color: #111827;">Upload File</h2>', unsafe_allow_html=True)
                
                # File Uploader
                uploaded_file = st.file_uploader(
                    f"Choose a {file_type.lower()}...", 
                    type=["jpg", "jpeg", "png", "mp4"]
                )
                
                # Preview Media
                if uploaded_file is not None:
                    st.success(f"File uploaded successfully: {uploaded_file.name}")
                    if file_type == "Image":
                        try:
                            image = Image.open(uploaded_file)
                            st.image(image, caption="Preview", width=400)
                        except Exception as e:
                            st.error("Error: Invalid Image File")
                    else:
                        st.video(uploaded_file)

                # Advanced Settings (Collapsible)
                with st.expander("Advanced Settings"):
                    st.markdown('<p style="color: #666; font-size: 0.9rem;">Fine-tune detection with custom models and thresholds.</p>', unsafe_allow_html=True)
                    
                    model = st.selectbox(
                        "Select Model", 
                        ("EfficientNetB4", "EfficientNetB4ST", "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
                    )
                    
                    col_set1, col_set2 = st.columns(2)
                    with col_set1:
                        dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
                    
                    with col_set2:
                        threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
                        if file_type == "Video":
                            frames = st.slider("Select Frames", 0, 100, 50)
                        else:
                            frames = 0 # Default for image

                # Check Button logic
                if uploaded_file is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Check for Deepfake", use_container_width=True):
                        # --- ORIGINAL LOGIC INTEGRATION START ---
                        with st.spinner("Analyzing frames and extracting features..."):
                            # This simulates the "Loading" page from HTML
                            time.sleep(1) # Small UX pause
                            
                            if file_type == "Image":
                                result, pred = process_image(
                                    image=uploaded_file, model=model, dataset=dataset, threshold=threshold)
                            else:
                                # Save video logic from original code
                                with open(f"uploads/{uploaded_file.name}", "wb") as f:
                                    f.write(uploaded_file.read())
                                
                                video_path = f"uploads/{uploaded_file.name}"
                                
                                result, pred = process_video(
                                    video_path, model=model, dataset=dataset, threshold=threshold, frames=frames)
                            
                            # Store results in session state to switch "view"
                            st.session_state.result_data = {
                                "result": result,
                                "pred": pred,
                                "file_type": file_type
                            }
                            st.rerun()
                        # --- ORIGINAL LOGIC INTEGRATION END ---

            # --- RESULTS SECTION (Shows after processing) ---
            else:
                data = st.session_state.result_data
                result_text = data["result"]
                probability = data["pred"]
                
                # Determine Colors and Icons based on result
                if result_text == 'fake':
                    color_class = "text-red-600"
                    bg_class = "bg-red-50 border-red-500"
                    icon = "fa-exclamation-triangle"
                    title = "Deepfake Detected"
                    desc = "This media appears to be manipulated."
                    progress_color = "#dc2626"
                else:
                    color_class = "text-green-600"
                    bg_class = "bg-green-50 border-green-500"
                    icon = "fa-check-circle"
                    title = "Authentic Media"
                    desc = "No manipulation detected."
                    progress_color = "#16a34a"

                # Render HTML Result Card
                st.markdown(f"""
                    <div class="mb-8 p-8 rounded-xl border-4 {bg_class}">
                        <div class="flex items-center justify-between mb-6">
                            <div class="flex items-center space-x-4">
                                <i class="fas {icon} text-6xl {color_class}"></i>
                                <div>
                                    <h2 class="text-3xl font-bold {color_class}">{title}</h2>
                                    <p class="text-lg text-gray-600">{desc}</p>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-5xl font-bold {color_class}">{probability:.2f}</div>
                                <div class="text-gray-600">Confidence Score</div>
                            </div>
                        </div>
                        
                        <div class="w-full bg-gray-200 rounded-full h-4 dark:bg-gray-700">
                          <div class="{bg_class.split()[0]} h-4 rounded-full" style="width: {probability*100}%; background-color: {progress_color}"></div>
                        </div>
                        <p style="text-align: center; margin-top: 10px; color: #666;">Probability: {int(probability*100)}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Action Buttons
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Analyze Another File"):
                        st.session_state.result_data = None
                        st.rerun()
                with c2:
                    st.button("Download Report (Demo)", disabled=True)

            # Close White Card
            st.markdown('</div>', unsafe_allow_html=True)

    </div>
    """, unsafe_allow_html=True)
