import threading
import time
from utils.gendern import gender_detector
from utils.age import age_detector

class prediction:
    def __init__(self, gen_det, age_det):
        self.age_detector = age_detector()
        self.gender_detector = gender_detector()
        self.images = []
        self.results = []
        self.gen_det=gen_det
        self.age_det=age_det

    def start_threads(self):
        x = threading.Thread(target=self.predict, args=([]), daemon=True)
        x.start()

    def pass_detections(self, dets):
        self.images = dets

    def predict(self):
        while True:
            try:
                newdets = []
                for img in self.images:
                    age=""
                    gender=""
                    # Predict gender
                    if(self.gen_det):
                        gender = self.gender_detector.detect_gender(img[1])
                    # Predict age
                    if(self.age_det):
                        age = self.age_detector.detect_age(img[1])
                    # Append results to newdets
                    newdets.append([img[0], img[1], gender, age])
                self.results = newdets
            except Exception as e:
                print(str(e))
            time.sleep(1/10)