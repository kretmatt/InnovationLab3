import cv2
import numpy as np

class gender_detector:
    def __init__(self, root):
        self.gender_model = cv2.dnn.readNetFromCaffe(str(root) + '/gender.prototxt', str(root) + '/gender.caffemodel')

    def detect_gender(self, img_face):
        detected_face = cv2.resize(img_face, (224, 224)) #img shape is (224, 224, 3) now
        img_blob = cv2.dnn.blobFromImage(detected_face) # img_blob shape is (1, 3, 224, 224)
        self.gender_model.setInput(img_blob)
        gender_class = self.gender_model.forward()[0]
        gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
        return gender
