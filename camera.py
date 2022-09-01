import cv2
from emotion import EmotionModel
import numpy as np

haar_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = EmotionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        self.video.release()

    # Grabs frame used for manipulation
    def get_frame(self):
        _, frame = self.video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts image to gray colour space
        faces = haar_casc.detectMultiScale(gray_frame, 1.3, 5) # Haar Cascades detect face

        # Loop to retrieve coordinates of face
        for (x, y, w, h) in faces:
            face_xy = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(face_xy, (48, 48)) # Resize to 48x48 to work with model
            emotion = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis]) # Predictions made in emotion file
            cv2.putText(frame, emotion, (x, y), font, 1, (180, 180, 0), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,0),2)

        _, jpeg = cv2.imencode('.jpg', frame) # Save image as a jpeg
        return jpeg.tobytes() # Return as bytes so Flask can display it
