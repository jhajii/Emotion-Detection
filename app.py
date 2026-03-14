import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os
from datetime import datetime

# Performance settings
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
cv2.setNumThreads(0)

# Fix for older keras models
import keras
@keras.saving.register_keras_serializable(package="Custom")
class PatchedDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        kwargs.pop("optional", None)
        super().__init__(**kwargs)

# Page setup
st.set_page_config(page_title="Emotion AI | Pritam", layout="centered")

# UI Styling
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
background-color: #1e2130;
border-radius: 8px;
color: white;
padding: 10px 15px;
}
.stTabs [aria-selected="true"] {
background-color: #ff4b4b !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Real-Time Emotion Recognition")

# File paths (based on your folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_detection.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Load resources
@st.cache_resource
def setup_resources():
    model = load_model(
        MODEL_PATH,
        custom_objects={"Dense": PatchedDense},
        compile=False
    )

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    return model, cascade

model, face_cascade = setup_resources()

emotion_labels = {
0: "Angry",
1: "Happy",
2: "Neutral",
3: "Sad",
4: "Surprised"
}

color_map = {
0: (0,0,255),
1: (0,255,0),
2: (255,255,255),
3: (255,0,0),
4: (0,255,255)
}

# Emotion detection function
def process_emotion(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(30,30)
    )

    for (x,y,w,h) in faces:

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))

        roi = roi.astype("float32") / 255.0
        roi = np.reshape(roi, (1,48,48,1))

        prediction = model.predict(roi, verbose=0)
        idx = np.argmax(prediction)

        label = emotion_labels[idx]
        color = color_map[idx]

        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        cv2.putText(
            image,
            label,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    return image, len(faces)

# Webcam processing
def video_callback(frame):

    img = frame.to_ndarray(format="bgr24")
    processed,_ = process_emotion(img)

    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# Tabs
tab1, tab2, tab3 = st.tabs([
"🏠 Home Info",
"🎥 Live Detection",
"📤 Upload Image"
])

# Home tab
with tab1:

    st.subheader("Emotion Recognition AI")

    st.write("""
This AI system detects facial emotions using
a deep learning CNN model.
""")

    st.markdown("---")

    st.subheader("Sample Images")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.image("dt1.jpeg")

    with col2:
        st.image("dt2.jpeg")

    with col3:
        st.image("dt3.jpeg")

# Live detection tab
with tab2:

    st.info("Allow camera access")

    webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_callback,
        media_stream_constraints={
        "video": True,
        "audio": False
        },
        async_processing=True
    )

# Upload tab
with tab3:

    file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if file:

        file_bytes = np.asarray(
            bytearray(file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes,1)

        with st.spinner("Analyzing Emotion..."):

            processed,count = process_emotion(img)

        st.image(
            cv2.cvtColor(processed,cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

        st.success(f"Detected {count} face(s)")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        _,buffer = cv2.imencode(".jpg", processed)

        st.download_button(
            "Download Result",
            data=buffer.tobytes(),
            file_name=f"emotion_{ts}.jpg"
        )
