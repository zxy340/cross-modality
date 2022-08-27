# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
from hashlib import new
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mm_process import process
from steaming import adcCapThread
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / './runs/train/Cross1-1_yolov3-tiny/weights/best.pt',  # model.pt path(s)
        source='../hand/hand/data/realtime/',  # file/dir/URL/glob, 0 for webcam
        imgsz=416,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Open a thread for mmWave
    # a = adcCapThread(1, "adc")
    # a.start()

    while True:
        t1 = time_sync()
        # Kinect Dataloader
        # typ = np.dtype((np.uint16, (424, 512)))
        # Depth_image = np.fromfile(source + 'Depdata.txt', dtype=typ)
        # Depth_image = Depth_image.squeeze()
        # if len(Depth_image) == 0:
        #     continue
        # max_value = Depth_image.max()
        # Depth_image = Depth_image / max_value * 255
        # Depth_image = cv2.resize(Depth_image, (416, 416))
        kin_name = './realtime/kin_image.jpg'
        # cv2.imwrite(kin_name, Depth_image)
        Depth_image = cv2.imread(kin_name)
        # mmWave Dataloader
        # readItem, _, _ = a.getFrame()
        # if len(readItem) == 14:
        #     continue
        frame = 1500
        x0 = np.fromfile('./realtime/adc_data_Raw_0.bin', dtype=np.int16)
        x1 = np.fromfile('./realtime/adc_data_Raw_1.bin', dtype=np.int16)
        x2 = np.fromfile('./realtime/adc_data_Raw_2.bin', dtype=np.int16)
        x = np.concatenate((x0, x1, x2), axis=0)
        x = x.reshape((frame, -1))
        readItem = process(x[200])
        t2 = time_sync()
        mm_image = process(readItem)
        t3 = time_sync()
        name = './realtime/mm_image.jpg'
        cv2.imwrite(name, mm_image)
        dataset = LoadImages(name, img_size=imgsz, stride=stride, auto=pt and not jit)

        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        seen, count =0, 0
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = model(im, augment=augment, visualize=visualize)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred[0] = pred[0][torch.arange(pred[0].size(0))==0]
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    count = count + 1  # record the number of detected images
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                # old_mtime = np.genfromtxt(source + 'timestamp.txt', dtype=str)
                # f = open(source + 'timestamp.txt')
                # old_mtime = f.readline()
                # f.close()
                # while True:
                results = np.hstack([im0, Depth_image])
                
                cv2.imshow(str(p), results)
                cv2.waitKey(1)  # 1 millisecond
                    # new_mtime = np.genfromtxt(source + 'timestamp.txt', dtype=str)
                    # f = open(source + 'timestamp.txt')
                    # new_mtime = f.readline()
                    # f.close()
                    # if new_mtime != old_mtime:
                    #     break
        t4 = time_sync()
        LOGGER.info(f'(load data)Done. ({t2 - t1:.3f}s)')
        LOGGER.info(f'(process mmWave data)Done. ({t3 - t2:.3f}s)')
        LOGGER.info(f'(model prediction)Done. ({t4 - t3:.3f}s)')

def main():
    run()

if __name__ == "__main__":
    main()
