import cv2
import numpy as np
import tensorflow as tf

class emotion_detector:

    def __init__(self):
        self.emotion_model = tf.keras.models.load_model('models/emotion_model.h5')
        self.emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "neutral", 4: "happy", 5: "sad", 6: "surprise"}

    def detect_emotion(self, img_face):
        try:
            gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            result = self.emotion_model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            emotion = self.emotion_dict[label]
            return emotion
        except Exception as e:
            print(str(e))
            return 'UNDETECTED'




