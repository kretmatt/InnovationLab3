import cv2
import numpy as np

class age_detector:

    def __init__(self, root):
        self.age_model = cv2.dnn.readNetFromCaffe(str(root) + '/deploy_age.prototxt', str(root) + '/age_net.caffemodel')

    def detect_age(self, img_face):
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        img_blob = cv2.dnn.blobFromImage(img_face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        self.age_model.setInput(img_blob)
        age_class = self.age_model.forward()
        age = ageList[age_class[0].argmax()]
        return age
