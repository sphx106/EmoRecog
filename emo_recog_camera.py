import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class EmoRecog(object):

    def __init__(self, model, img_height, img_width, face_detector, emotion_mapping):
        self.model = model
        self.img_height = img_height
        self.img_width = img_width
        self.face_detector = face_detector
        self.emotion_mapping = emotion_mapping

    def get_emotion_box(self, frame):
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(grayscale_frame, 1.3, 5)
        x, y, w, h = faces[0]
        face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, dsize=(self.img_height, self.img_width), interpolation=cv2.INTER_CUBIC)
        face = np.expand_dims(face, axis=0)
        pred = self.model.predict(face)
        emotion_index = np.argmax(pred, axis=1)
        emotion = [self.emotion_mapping[i] for i in emotion_index]
        return emotion, faces[0]
    
    def get_frame_emotion(self, frame):
        emotion, box = self.get_emotion_box(frame)
        x, y, w, h = box
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        frame = cv2.putText(frame, emotion[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame, emotion
    
IMG_HEIGHT = 200
IMG_WIDTH = 200
MODEL_PATH = 'code/model_norm_aug_custom_2_15'

face_model = tf.keras.models.load_model(MODEL_PATH)

emotion_mapping = {0: 'anger',
1: 'contempt',
2: 'disgust',
3: 'fear',
4: 'happy',
5: 'neutral',
6: 'sad',
7: 'surprise',
8: 'uncertain'}

face_detector = cv2.CascadeClassifier('code/haarcascade_frontalface_default.xml')
emotion_recognition_tool = EmoRecog(face_model, 
                                    IMG_HEIGHT, 
                                    IMG_WIDTH, 
                                    face_detector, 
                                    emotion_mapping)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while(True):
    ret, frame = cam.read()
    cv2.imshow("facial emotion recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        cv2.imwrite('code/frames/emotion_frame.jpg', frame)
        break

frame = cv2.imread('code/frames/emotion_frame.jpg')
  
frame, emotion = emotion_recognition_tool.get_frame_emotion(frame)
print(emotion[0])
cv2.imshow("facial emotion recognition", frame)
cv2.waitKey(0)