import json
import cv2
import base64
import numpy as np
import requests
import time

import paho.mqtt.client as mqtt #import MQTT Client

broker_address="broker.hivemq.com" 
port=1883

client = mqtt.Client("exer1") #create new instance
client.connect(broker_address, port) #connect to MQTT broker


# load config
with open('roboflow_config_e.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    ROBOFLOW_MODEL_ID = config["ROBOFLOW_MODEL_ID"]
    ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL_ID, "/",
    ROBOFLOW_VERSION_NUMBER,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=json",
    "&confidence=80",
    "&stroke=5"
])

video = cv2.VideoCapture(0)

def infer():
    
    ret, img = video.read()

    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    
    img = cv2.resize(img, (round(scale * width), round(scale * height)))
    
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
   
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True)
    
    predictions = resp.json()
    detections = predictions['predictions']

    for bounding_box in detections:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2
        class_name = bounding_box['class']
        confidence = bounding_box['confidence']

        print(class_name)

        client.publish("estatus", class_name) # publish detected class to MQTT Server
        

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
    
        cv2.rectangle(img, start_point, end_point, color=(0,0,255), thickness=1)

        (text_width, text_height), _ = cv2.getTextSize(
            f"{class_name}",
        
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)

        cv2.rectangle(img, (int(x0), int(y0)), (int(x0) + text_width, int(y0) - text_height), color=(0,0,255),
            thickness=-1)
        
        text_location = (int(x0), int(y0))
             
        cv2.putText(img, f"{class_name}",
                    text_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255,255,255), thickness=1) 
                              
    return img, detections

while 1:
   
    if(cv2.waitKey(1) == ord('q')):
        break

    start = time.time()
    image, detections = infer()
    cv2.imshow('image', image)
    print((1/(time.time()-start)), " fps")
    print(detections)
    
video.release()
cv2.destroyAllWindows()