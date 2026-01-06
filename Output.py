import streamlit as st
from PIL import Image
from api import process_image, process_video

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# PROFESSIONAL UI STYLING (UI ONLY)
# --------------------------------------------------
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

.stApp {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    font-family: 'Inter', sans-serif;
}

/* Center container */
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ---------------- HERO SECTION ---------------- */
.hero-wrapper {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero-box {
    text-align: center;
    padding: 60px;
    background: rgba(255,255,255,0.08);
    border-radius: 24px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.35);
    max-width: 900px;
}

.hero-title {
    font-size: 72px;
    font-weight: 900;
    color: white;
    margin-bottom: 20px;
}

.hero-subtitle {
    font-size: 34px;
    font-weight: 600;
    color: rgba(255,255,255,0.95);
    margin-bottom: 28px;
}

.hero-desc {
    font-size: 22px;
    color: rgba(255,255,255,0.9);
    line-height: 1.7;
    margin-bottom: 45px;
}

/* ---------------- MAIN CARD ---------------- */
.main-container {
    background: white;
    border-radius: 22px;
    padding: 48px;
    box-shadow: 0 25px 70px rgba(0,0,0,0.25);
}

/* ---------------- TEXT SIZES ---------------- */
.stMarkdown h1 {
    font-size: 44px !important;
    font-weight: 800 !important;
}

.stMarkdown h2 {
    font-size: 34px !important;
    font-weight: 700 !important;
}

.stMarkdown h3 {
    font-size: 26px !important;
    font-weight: 600 !important;
}

/* ---------------- BUTTONS ---------------- */
.stButton > button {
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 18px 48px !important;
    border-radius: 14px !important;
    min-height: 64px !important;
    transition: all 0.3s ease !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 10px 25px rgba(239,68,68,0.45) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-3px);
}

/* ---------------- INPUTS ---------------- */
.stRadio label, .stSelectbox label, .stSlider label {
    font-size: 20px !important;
    font-weight: 600 !important;
}

.stFileUploader {
    font-size: 18px !important;
}

/* ---------------- RESULT ---------------- */
.result-text {
    font-size: 30px;
    font-weight: 800;
    margin-top: 30px;
}

/* ---------------- RESPONSIVE ---------------- */
@media (max-width: 768px) {
    .hero-title { font-size: 48px; }
    .hero-subtitle { font-size: 26px; }
    .hero-desc { font-size: 18px; }
    .main-container { padding: 30px 22px; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "hero"

# --------------------------------------------------
# HERO PAGE (INTRO PAGE)
# --------------------------------------------------
if st.session_state.page == "hero":
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)

    st.markdown('<div class="hero-title">Deepfake Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Detect AI-generated images and videos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-desc">'
        'Upload an image or video to check if it has been manipulated using deepfake techniques.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APP PAGE (UNCHANGED LOGIC)
# --------------------------------------------------
else:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.title("Deepfake Detector App")

    file_type = st.radio("Select file type:", ("Image", "Video"))

    uploaded_file = st.file_uploader(
        f"Choose a {file_type.lower()}...",
        type=["jpg", "jpeg", "png", "mp4"]
    )

    model = st.selectbox(
        "Select Model",
        ("EfficientNetB4", "EfficientNetB4ST",
         "EfficientNetAutoAttB4", "EfficientNetAutoAttB4ST")
    )

    dataset = st.radio("Select Dataset", ("DFDC", "FFPP"))
    threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)

    if file_type == "Video":
        frames = st.slider("Select Frames", 0, 100, 50)

    if uploaded_file is not None:
        if file_type == "Image":
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=260)
            except:
                st.error("Invalid image file.")
        else:
            st.video(uploaded_file)

        if st.button("Check for Deepfake", type="primary", use_container_width=True):
            if file_type == "Image":
                result, pred = process_image(
                    image=uploaded_file,
                    model=model,
                    dataset=dataset,
                    threshold=threshold
                )
            else:
                with open(f"uploads/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.read())

                result, pred = process_video(
                    f"uploads/{uploaded_file.name}",
                    model=model,
                    dataset=dataset,
                    threshold=threshold,
                    frames=frames
                )

            st.markdown(
                f"""
                <h3 class="result-text">
                The given {file_type} is:
                <span style="color:{'#ef4444' if result=='fake' else '#22c55e'}">
                {result.upper()}
                </span>
                <br>
                Probability: <b>{pred:.2f}</b>
                </h3>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Please upload a file.")

    st.markdown('</div>', unsafe_allow_html=True)
