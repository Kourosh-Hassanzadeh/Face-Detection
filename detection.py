import cv2
import logging 
from time import sleep
import datetime as dt

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
logging.basicConfig(filename='webcam.log', level=logging.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print("Unable to load camera.")
        sleep(5)
        pass
        
    rec, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text_position = (x, y - 10)
        cv2.putText(frame, 'face', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if anterior != len(faces):
        anterior = len(faces)
        logging.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cv2.imshow('Video', frame)
        
video_capture.release()
cv2.destroyAllWindows()