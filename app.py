import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
from datetime import datetime

# Performance optimizations
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
cv2.setNumThreads(0)

# Keras Patch for Legacy Models
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        kwargs.pop('optional', None)
        super().__init__(**kwargs)

st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# --- UI Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130; border-radius: 8px; color: white; padding: 10px 15px;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; }
    div[data-testid="stSidebarNav"] { display: none; }
    .info-card { background-color: #1e2130; padding: 25px; border-radius: 15px; border-left: 5px solid #ff4b4b; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def setup_resources():
    model = load_model("emotion_detection.h5", custom_objects={"Dense": PatchedDense}, compile=False)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return model, cascade

model, face_cascade = setup_resources()
emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
color_map = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 255, 255), 3: (255, 0, 0), 4: (0, 255, 255)}

# --- Preprocessing & Detection Engine ---
def process_emotion(image):
    h, w = image.shape[:2]
    max_side = 900 
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Accuracy Boost: Advanced Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Face Detection Logic
    faces = face_cascade.detectMultiScale(
        gray_enhanced, 
        scaleFactor=1.1, 
        minNeighbors=6, 
        minSize=(30, 30)
    )
    
    if len(faces) > 15:
        return "limit", len(faces)
    
    for (x, y, fw, fh) in faces:
        # Preprocessing ROI for Model Accuracy
        roi_gray = gray_enhanced[y:y+fh, x:x+fw]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Noise Reduction & Normalization
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        
        # Prediction
        prediction = model.predict(roi_gray, verbose=0)
        idx = np.argmax(prediction)
        
        # Confidence Check (Optional but good for quality)
        label = emotion_labels[idx]
        color = color_map[idx]
        
        thickness = max(2, int(w / 500))
        font_scale = max(0.45, w / 1000)
        
        text_y = y - 10 if y - 10 > 25 else y + fh + 30
        cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(image, label, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        cv2.rectangle(image, (x, y), (x+fw, y+fh), color, thickness)
        
    return image, len(faces)

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    res = process_emotion(img)
    processed_img = res[0] if isinstance(res, tuple) else img
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- UI Structure ---
tab_info, tab_live, tab_upload = st.tabs(["🏠 Home Info", "🎥 Live Detection", "📤 Upload Image"])

with tab_info:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    with col2:
        st.subheader("Emotion Recognition AI")
        st.write("A professional-grade Deep Learning system designed to interpret facial micro-expressions in real-time.")
    
    st.markdown("---")
    st.subheader("System Overview")
    st.markdown("""
    - **Architecture:** Convolutional Neural Network (CNN) optimized for FER2013.
    - **Face Tracking:** Robust Haar-Cascade multi-face detection (optimized for groups).
    - **Image Processing:** CLAHE contrast enhancement and Gaussian noise reduction.
    - **Stability:** Dynamic rescaling for high-resolution input handling.
    """)
    
    st.markdown("---")
    st.subheader("📬 Feedback & Contact")
    st.info("I am constantly working to improve this model's accuracy. If you notice any incorrect detections, please share the result with me via the links below. Your feedback is invaluable!")
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/pritam-kumar-607631334)")
    c2.markdown("[📸 Instagram](https://www.instagram.com/pritamray26)")
    c3.markdown("[📧 Email](mailto:pritamray6200@gmail.com)")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_live:
    st.info("Directly accessing front camera. Best used in well-lit conditions.")
    webrtc_streamer(
        key="emotion-live-accurate",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": {"facingMode": "user", "width": {"ideal": 640}}, "audio": False},
        async_processing=True,
    )

with tab_upload:
    file = st.file_uploader("Upload Image (Max 15 faces)", type=['jpg', 'png', 'jpeg'])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Applying Deep Preprocessing & Analysis...'):
            result = process_emotion(img)
            
            if result == "limit":
                st.error("Too many faces! Please upload an image with up to 15 members.")
            else:
                processed_img, count = result
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.success(f"Analysis complete! Detected {count} face(s).")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _, enc = cv2.imencode('.jpg', processed_img)
                st.download_button("📥 Save Analysis", data=enc.tobytes(), file_name=f"emotion_{ts}.jpg")
