import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

#Read a model
model = YOLO('yolov8n.pt')

#Input
#camera_img = st.camera_input(label='InCameraImage')
camera_img = st.file_uploader("Uppload a Image", type = ['png','jpg', 'jpeg'])

#Process
if camera_img is not None:

  conf_input = st.slider("Confidence", min_value = 0.0, max_value = 1.0, value = 0.50, step = 0.01)
  option_person = st.radio("Detect Only Person?", ["Only Person", "All Objects"])
  bytes_data = camera_img.getvalue() #Digitalize Image in BGR
  cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR) #change BGR to RGB
  if option_person == "Only Person":
    result = model(cv2_img, conf = conf_input, classes = [0]) #unless designate classes, it detect 0-79 objects
  else :
    result = model(cv2_img, conf = conf_input)
  output_img = result[0].plot(labels=True, conf=True)
  output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

#Output
  st.image(output_img, caption = 'Output Image')
  if option_person == "Only Person":
    st.text(f'There are {len(result[0].boxes.cls)} people')