import cv2
import numpy as np
import tensorflow as tf


class gender_detector:
    def __init__(self):
        self.gender_modelK = tf.keras.models.load_model('models/gender_model.h5')
        self.labels_dict = {0: 'Male', 1: 'Female'}

    def detect_gender(self, img_face):
        try:
            # Gender Detection
            resized = cv2.resize(img_face, (178, 218))
            reshaped = np.reshape(resized, (1, 178, 218, 3))
            result = self.gender_modelK.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            return self.labels_dict[label];
        except Exception as e:
            print(str(e))
