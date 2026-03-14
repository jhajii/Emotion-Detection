import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import os
import gdown
from datetime import datetime

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
cv2.setNumThreads(0)

# Download model if not present
MODEL_URL = "https://drive.google.com/uc?id=1qJX9K0exampleMODELID"   # replace with your drive link
MODEL_PATH = "emotion_detection.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

CASCADE_PATH = "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("🎭 Real-Time Emotion Recognition")

@st.cache_resource
def setup_resources():
    model = load_model(MODEL_PATH, compile=False)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return model, cascade

model, face_cascade = setup_resources()

emotion_labels = {
0:"Angry",
1:"Happy",
2:"Neutral",
3:"Sad",
4:"Surprised"
}

color_map = {
0:(0,0,255),
1:(0,255,0),
2:(255,255,255),
3:(255,0,0),
4:(0,255,255)
}

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
        roi = cv2.resize(roi,(48,48))
        roi = roi.astype("float32")/255.0
        roi = np.reshape(roi,(1,48,48,1))

        pred = model.predict(roi,verbose=0)
        idx = np.argmax(pred)

        label = emotion_labels[idx]
        color = color_map[idx]

        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        cv2.putText(image,label,(x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    return image,len(faces)

def video_callback(frame):

    img = frame.to_ndarray(format="bgr24")
    processed,_ = process_emotion(img)

    return av.VideoFrame.from_ndarray(processed,format="bgr24")

tab1,tab2,tab3 = st.tabs(["Info","Live","Upload"])

with tab1:

    st.write("Deep learning based facial emotion recognition.")

with tab2:

    webrtc_streamer(
        key="emotion",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_callback,
        media_stream_constraints={
        "video":True,
        "audio":False
        },
        async_processing=True
    )

with tab3:

    file = st.file_uploader("Upload image",type=["jpg","jpeg","png"])

    if file:

        bytes_data = np.asarray(bytearray(file.read()),dtype=np.uint8)
        img = cv2.imdecode(bytes_data,1)

        processed,count = process_emotion(img)

        st.image(cv2.cvtColor(processed,cv2.COLOR_BGR2RGB))
        st.success(f"{count} face detected")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        _,buffer = cv2.imencode(".jpg",processed)

        st.download_button(
        "Download",
        data=buffer.tobytes(),
        file_name=f"emotion_{ts}.jpg"
        )
