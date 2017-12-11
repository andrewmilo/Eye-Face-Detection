import cv2
import sys
import datetime as dt
from time import sleep
from datetime import datetime, time

facePath = "haarcascade_frontalface_default.xml"
eyePath = "haarcascade_eye.xml"
faces_cascade = cv2.CascadeClassifier(facePath)
eyes_cascade = cv2.CascadeClassifier(eyePath)
video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faces_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    eyes = eyes_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # if len(eyes) == 2:
    #     print "Hey YOU!"

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw rectangles over eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)

    resized_image = cv2.resize(frame, (1366,768))

    # Display the resulting frame
    cv2.imshow('Video', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', resized_image)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
