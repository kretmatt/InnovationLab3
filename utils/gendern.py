import cv2
import numpy as np
import keras
import tensorflow as tf

class gender_detector:
    def __init__(self, root):
        #self.gender_model = cv2.dnn.readNetFromCaffe(str(root) + '/deploy_gender.prototxt', str(root) + '/gender_net.caffemodel')
        self.gender_modelK = tf.keras.models.load_model('gender_model.h5')

    def detect_gender(self, img_face):
        try:
            labels_dict = {0: 'Male', 1: 'Female'}
            gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32))
            normalized = resized/255.0
            reshaped = np.reshape(normalized, (1, 32, 32, 1))
            result = self.gender_modelK.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            #detected_face = cv2.resize(img_face, (227, 227)) #img shape is (224, 224, 3) now
            #img_blob = cv2.dnn.blobFromImage(detected_face) # img_blob shape is (1, 3, 224, 224)
            #self.gender_model.setInput(img_blob)
            #gender_class = self.gender_model.forward()[0]
            #gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
            #return gender
            return labels_dict[label]
        except Exception as e:
            print(str(e))
            return 'UNDETECTED'
