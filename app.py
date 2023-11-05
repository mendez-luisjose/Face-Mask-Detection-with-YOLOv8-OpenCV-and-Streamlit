#Importing the Modules Needed
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from utils import set_background
import tempfile
import time 

set_background("./assets/background.png")

FACE_MASK_MODEL_PATH = './model/face_mask_model.pt'
DEMO_VIDEO_PATH = "./test_images/video_test.mp4"

header = st.container()
body = st.container()

#Loading the Face Mask Detection Model
model = YOLO(FACE_MASK_MODEL_PATH)

threshold = 0.30

img = None

state = "Uploader"

#Model Prediction Function
def model_prediction(img, threshold):
    counter = 0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred = model.predict(img)[0]

    for result in pred.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            img = pred[0].plot()

            counter+=1
            
    img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_wth_box, counter

#Image Resize Function
@st.cache_data()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA) :
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None :
        return image
    
    if width is None: 
        r = width /  float(w)
        dim = (int(w * r), height)
    else:
        r = width/float(w)
        dim = (width, int(h *r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


#App made with Streamlit

st.sidebar.markdown('''
    🧑🏻‍💻 Created by [Luis Jose Mendez](https://github.com/mendez-luisjose).
    ''')

app_mode = st.sidebar.selectbox("Select App Mode: ",
                                ["Upload Photo", "Take Photo", "Live Detection"])    

if app_mode == "Upload Photo" :
    st.sidebar.markdown("---------")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.subheader("Parameters:")
    detection_confidence = st.sidebar.slider("Min Detection Confidence:", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown("---------")

    img_file = st.sidebar.file_uploader("Upload Image:", type=["jpg", "png", "jpeg"])

    if img_file is not None :
        image = np.array(Image.open(img_file))
    else :
        demo_image = "./test_images/test.png"
        image = np.array(Image.open(demo_image))

    st.sidebar.text("Original Image:")
    st.sidebar.image(image)

    if st.sidebar.button("Apply Detection") :

        output_image, mask_counter = model_prediction(image, detection_confidence)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.title("Output Image ✅:")
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.image(output_image, use_column_width=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        _, col2, _ = st.columns([0.5, 1, 0.5])

        kpil_text = col2.markdown("0")
        kpil_text.write(f"<h1 style='text-align: center; color: red'><p style='font-size: 0.7em; color:white'>Detected Face Mask:</p>{mask_counter}</h1>", unsafe_allow_html=True)

elif app_mode == "Take Photo" :
    st.sidebar.markdown("---------")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.subheader("Parameters:")
    detection_confidence = st.sidebar.slider("Min Detection Confidence:", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown("---------")

    img_file = st.sidebar.camera_input("Take a Photo: ")

    if img_file is not None :
        image = np.array(Image.open(img_file))
    else :
        demo_image = "./test_images/test.png"
        image = np.array(Image.open(demo_image))


    st.sidebar.text("Original Image:")
    st.sidebar.image(image)

    if st.sidebar.button("Apply Detection") :

        output_image, mask_counter = model_prediction(image, detection_confidence)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.title("Output Image ✅:")
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.image(output_image, use_column_width=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        _, col2, _ = st.columns([0.5, 1, 0.5])

        kpil_text = col2.markdown("0")
        kpil_text.write(f"<h1 style='text-align: center; color: red'><p style='font-size: 0.7em; color:white'>Detected Face Mask:</p>{mask_counter}</h1>", unsafe_allow_html=True)

elif app_mode == "Live Detection" :
    st.set_option("deprecation.showfileUploaderEncoding", False)

    use_webcam = st.sidebar.button("Use Webcam")

    st.sidebar.markdown("---------")

    st.sidebar.subheader("Parameters:")
    detection_confidence = st.sidebar.slider("Min Detection Confidence:", min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown("---------")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.title("Output Video ✅:")
    st.markdown("<hr/>", unsafe_allow_html=True)

    st_frame = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video:", type=["mp4", "mov", "avi", "asf", "m4v"])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    if not video_file_buffer:
        if use_webcam:
            video = cv2.VideoCapture(0)

            _, col, _ = st.columns([0.8, 1, 0.5])
            col.checkbox("Recording Live Detection", value=True)
            st.markdown("<hr/>", unsafe_allow_html=True)
        else:
            pass
            video = cv2.VideoCapture(DEMO_VIDEO_PATH)
            st.markdown("<hr/>", unsafe_allow_html=True)
            tffile.name = DEMO_VIDEO_PATH    
    else :
        tffile.write(video_file_buffer.read())
        video = cv2.VideoCapture(tffile.name)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter("output1.mp4", codec, fps_input, (width, height))

    st.sidebar.text("Input Video:")
    st.sidebar.video(tffile.name)

    col1, col2 = st.columns([0.8, 1])

    col1_text = col1.markdown("0")
    col2_text = col2.markdown("0")

    FPS = 0
    i = 0
    prevTime = 0

    while video.isOpened() :
        i+=1
        ret, frame = video.read()

        if not ret:
            continue
        
        frame.flags.writeable = True    

        face_count = 0
    
        #FPS Counter Logic
        currentTime = time.time()
        FPS = 1 / (currentTime - prevTime)   
        prevTime = currentTime       

        frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
        frame = image_resize(image = frame, width = 640)

        output_image, mask_counter = model_prediction(frame, detection_confidence)

        st_frame.image(output_image, channels = "BGR", use_column_width = True)

        col1_text.write(f"<h1 style='text-align: center; color: red'><p style='font-size: 0.7em; color:white'>FPS:</p>{int(FPS)}</h1>", unsafe_allow_html=True)
        col2_text.write(f"<h1 style='text-align: center; color: red'><p style='font-size: 0.7em; color:white'>Face Mask Detected:</p>{mask_counter}</h1>", unsafe_allow_html=True)

with header :
    _, col1, _ = st.columns([0.25,1,0.1])
    col1.title("Face Mask Detection 😷")

    st.markdown("<hr/>", unsafe_allow_html=True)

    _, col2, _ = st.columns([0.05,1,0.1])
    col2.image("./test_images/test-2.jpg")

    st.markdown("<hr/>", unsafe_allow_html=True)
    
with body :
    _, col1, _ = st.columns([0.25,1,0.2])
    col1.subheader("Computer Vision Project with YoloV8 🧪")

    st.write("The Model was trained with the Yolov8 Architecture, for 100 epochs, using the Google Colab GPU, and with more than 2000 Images.")







 