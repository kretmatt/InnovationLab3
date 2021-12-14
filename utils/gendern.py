import cv2
import numpy as np
import keras
import tensorflow as tf
import threading
import time

class gender_detector:
    def __init__(self, root):
        self.gender_modelK = tf.keras.models.load_model('gender_model.h5')
        self.images = []
        self.results = []
        x = threading.Thread(target=self.detect_gender, args=([]), daemon=True)
        x.start()


    def pass_detections(self, dets):
        self.images = dets

    def detect_gender(self):
        labels_dict = {0: 'Male', 1: 'Female'}
        print("asd")
        while True:
            try:
                newdets = []
                for img in self.images:
                    gray = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (32, 32))
                    normalized = resized/255.0
                    reshaped = np.reshape(normalized, (1, 32, 32, 1))
                    result = self.gender_modelK.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]
                    newdets.append([img[0], img[1], labels_dict[label]])
                self.results = newdets
            except Exception as e:
                print(str(e))