import cv2
import numpy as np
import tensorflow as tf

class age_detector:

    def __init__(self, root):
        #self.age_model = cv2.dnn.readNetFromCaffe(str(root) + '/deploy_age.prototxt', str(root) + '/age_net.caffemodel')
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

    def detect_age(self, img_face):
        try:
            """
            MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            img_blob = cv2.dnn.blobFromImage(img_face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            self.age_model.setInput(img_blob)
            age_class = self.age_model.forward()
            age = ageList[age_class[0].argmax()]
            """
            gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 32, 32, 1))
            result = self.age_model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            age = self.label_to_age_map[label]
            return age
        except Exception as e:
            print(str(e))
            return 'UNDETECTED'




