import cv2
import numpy as np
import tensorflow as tf


class gender_detector:
    def __init__(self):
        self.gender_modelK = tf.keras.models.load_model('models/gender_model_new.h5')
        self.labels_dict = {0: 'Male', 1: 'Female'}

    def detect_gender(self, img_face):
        try:
            # Gender Detection
            gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            resized = resized/255
            reshaped = np.reshape(resized, (1, 64, 64, 1))


            result = self.gender_modelK.predict(reshaped)
            #old code below
            #label = np.argmax(result[1], axis=1)[0]
            #return self.labels_dict[label];
            if(result[1] < 0.5):
                return 'Male'
            else:
                return 'Female'
        except Exception as e:
            print(str(e))
