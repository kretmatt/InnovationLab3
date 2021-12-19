import cv2
import numpy as np
import tensorflow as tf
import threading
import time

class gender_detector:
    def __init__(self, root):
        self.gender_modelK = tf.keras.models.load_model('gender_model.h5')
        self.age_model = tf.keras.models.load_model('age_model.h5')
        self.label_to_age_map = {
            0: '0-2',
            1: '4-6',
            2: '8-13',
            3: '15-20',
            4: '25-32',
            5: '38-43',
            6: '48-53',
            7: '60+'
        }
        self.images = []
        self.results = []
        x = threading.Thread(target=self.detect_gender, args=([]), daemon=True)
        x.start()


    def pass_detections(self, dets):
        self.images = dets

    def detect_gender(self):
        labels_dict = {0: 'Male', 1: 'Female'}
        print("starting gender thread")
        time.sleep(10)
        while True:
            try:
                newdets = []
                for img in self.images:
                    # Gender Detection
                    resized = cv2.resize(img[1], (178, 218))
                    reshaped = np.reshape(resized, (1, 178, 218, 3))
                    result = self.gender_modelK.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]

                    # Age Detection
                    gray = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
                    resized_age = cv2.resize(gray, (32, 32))
                    normalized = resized_age / 255.0
                    reshaped_age = np.reshape(normalized, (1, 32, 32, 1))
                    result_age = self.age_model.predict(reshaped_age)
                    label_age = np.argmax(result_age, axis=1)[0]

                    # Append results to newdets
                    newdets.append([img[0], img[1], labels_dict[label], self.label_to_age_map[label_age]])
                self.results = newdets
            except Exception as e:
                print(str(e))
            print("looping")
            time.sleep(1/10)