import threading
import time


class detection:
    def __init__(self):
        self.current_img = []


    def start_threads(self):
        x = threading.Thread(target=self.predict, args=([]), daemon=True)
        x.start()

    def detect(self):
        pass