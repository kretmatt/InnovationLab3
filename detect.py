#TODO: Documentation


import argparse
import os
import sys
import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
import threading

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.prediction import prediction

class inn3_detector:
    def __init__(self, weights=ROOT / 'models/face_detection_yolov5s.pt',  # model.pt path(s)
        source='0',  # file/dir/URL/glob, 0 for webcam
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        gen_det=False,
        age_det=False,
        emo_det=False
        ):
        self.source = str(source)
        self.predictor = prediction(gen_det, age_det, emo_det)
        self.device = select_device("")
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.current_pic=np.array([])

        # Load model
        self.stride, self.names, self.pt, self.jit, self.onnx = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        self.imgsz = check_img_size(640, s=self.stride)  # check image size

        # Dataloader
        self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt and not self.jit)

    def set_agedet(self, age_det):
        self.predictor.age_det=age_det

    def set_gendet(self, gen_det):
        self.predictor.gen_det=gen_det

    def set_emodet(self, emo_det):
        self.predictor.emo_det=emo_det

    def start_pred(self):
        self.predictor.start_threads()

    def get_currentpic(self):
        return self.current_pic

    def start_threads(self):
        x = threading.Thread(target=self.run, args=([]), daemon=True)
        x.start()

    @torch.no_grad()
    def run(self):
        # Run inference
        for path, im, im0s in self.dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=1000)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count

                p = Path(p)  # to Path
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det): #check if detection is not empty
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    crop_ims = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        crop_img = im0[(int(xyxy[1])):(int(xyxy[3])),(int(xyxy[0])):(int(xyxy[2]))] # Almost no impact on FPS counter
                        crop_ims.append([xyxy, crop_img])
                    self.predictor.pass_detections(crop_ims.copy())
                    pres = self.predictor.results.copy()
                    for res in pres:
                        box_col = 8 # default color
                        boxtext = res[2] + " " + res[3] + " " + res[4]
                        if(res[2]  == "Male"):
                            #print(gender)
                            box_col = 0
                        elif(res[2] == "Female"):
                            #print(gender)
                            box_col = 4
                        else:
                            #print("nothing detected")
                            box_col = 8
                        annotator.box_label(res[0], boxtext, color=colors(box_col, True))
                # Stream results
                # combine fps and conf

                #cv2.putText(im0, fps_conf_ctnr, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255),2)  # put fps counter on the top left corner
                im0 = annotator.result()
                #if view_img:
                self.current_pic=im0




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/face_detection_yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--gen_det', type=bool, default=False, help='gender detection, default false')
    parser.add_argument('--age_det', type=bool, default=False, help='age detection, default false')
    parser.add_argument('--emo_det', type=bool, default=False, help='emotion detection, default false')
    opt = parser.parse_args()
    #opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    print(vars(opt))
    idet = inn3_detector(**vars(opt))
    idet.start_pred()
    idet.start_threads()
    time.sleep(5)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
