import os
import hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import random
import math
import contextlib
import inspect
import urllib
import platform
import logging
import logging.config
import torch.distributed as dist
import yaml
import sys
import glob
import time

import pkg_resources as pkg

import cv2
import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageDraw, ImageOps, ImageFont
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from copy import deepcopy

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
LOGGING_NAME = 'chang'
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
# if platform.system() == 'Windows':
#     for fn in LOGGER.info, LOGGER.warning:
#         setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]
    
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s
    
def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def process_batch(detections,classes, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == classes
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def xyxy2xywhn(x, w, h, clip=False, eps=0.0):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
#     y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
#     y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
#     y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
#     return y


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def scale_coords(pad_par, gain, boxes, shape):

    if boxes.size == 0:
        return np.array([])
    boxes[:, :] *= gain

    if pad_par == None:
        pass
    else:
        boxes[:, [0, 2]] -= pad_par[1]
        boxes[:, [1, 3]] -= pad_par[0]
    for i, _ in enumerate(boxes):
        if _[0] < 0:
            boxes[i][0] = 0
        if _[1] < 0:
            boxes[i][1] = 0
        if _[2] > shape[1]:
            boxes[i][2] = shape[1]
        if _[3] > shape[0]:
            boxes[i][3] = shape[0]

    return boxes


def easystream_process(input, mask, anchors, imgz):
    # input.shape (80,80,3,11)
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    # input = sigmoid(input) #对 input 应用 Sigmoid 函数，特别是对边界框坐标和置信度进行尺度调整，使其值在 [0, 1] 之间
    box_confidence = input[..., 4] 

    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]
    box_xy = input[..., :2]*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)#(80,80,3,1)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)#(80,80,3,1)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid  #(80, 80, 3, 2) + (80, 80, 3, 2)
    box_xy *= int(imgz/grid_h) 

    # box_wh = pow(sigmoid(input[..., 2:4])*2, 2)

    box_wh = pow(input[..., 2:4]*2, 2)
    # (80,80,3,2) (3,2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return np.concatenate((box, box_confidence, box_class_probs), axis=3)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def easystream_post_process(input, Order, anchors_name, imgz,nc,conf_thres=0.45, iou_thres=0.65,maxdets=100,maxbboxs=1024):
    input0_data = input[Order[2]] #input[0] (1,33,80,80)
    input1_data = input[Order[1]] 
    input2_data = input[Order[0]]

    input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:])) #(3,11,80,80)
    input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1))) #(80,80,3,11)
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    boxes, classes, scores = [], [], []
    z = []
    for input, mask in zip(input_data, masks):
        x = np.transpose(
            easystream_process(input, mask, anchors_name, imgz), (2, 0, 1, 3))
        z.append(np.reshape(x, (1, -1, 5+nc)))

    pred = torch.from_numpy(np.concatenate(z, axis=1))

    # #dump txt befor nms
    # with open("bbox.txt","w",encoding="utf-8") as f:
    #     for i in z:
    #         for j in i[0]:
    #             if j[4]>=conf_thres:
    #                 probs=j[4:]
    #                 id=np.argmax(probs)-1
    #                 f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.6f} {}\n".format(j[0],j[1],j[2],j[3],j[4],id))

    pred = non_max_suppression(pred, conf_thres, iou_thres,maxdets,maxbboxs)[0].numpy()

    boxes = pred[:, :4]
    classes = pred[:, -1]
    scores = pred[:, 4]

    return boxes, classes, scores

def easystream_draw(image, boxes, scores, classes, CLASSES, line_thickness=3):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        CLASSES: all classes name.
    """
    if boxes.size == 0:
        return
    for box, score, cl in zip(boxes, scores, classes):
       # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        color = [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        label = "{} {:.2f}".format(CLASSES[int(cl)], score)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(
            label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def save_txt(img0, boxes, scores, classes, f):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    if boxes.size == 0:
        return
    boxes = xyxy2xywhn(boxes, img0.shape[1], img0.shape[0])
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        f.write("{} {} {} {} {} {}\n".format(int(cl), x, y, w, h, score))


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,maxdets=100,maxbboxs=1024,classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = maxdets  # maximum number of detections per image
    max_nms = maxbboxs  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    loop=0
    t = time.time()
    output = [torch.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            x[:, 5:] = x[:, 4:5]
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # sort by confidence
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        print(len(boxes))
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        loop=loop+1
        print(loop,len(output[xi]))
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

# self.anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
def _make_grid(na,stride, nx=20, ny=20, i=0):
    rgbt_anchors=[[[1.9668, 4.1328],[8.7656, 3.3008],[3.2109, 9.7422]],
                  [[2.4336, 6.5547],[3.4062, 10.2109],[7.3984, 5.0312]],
                  [[7.3750, 3.2832],[5.2305, 5.1953],[3.7168, 10.0938]]]
    # rgbt_anchors=np.array(rgbt_anchors) 
    rgbt_anchors=torch.tensor(rgbt_anchors)              
    shape = 1, na, ny, nx, 2  # grid shape
    # 生成 y 和 x 的坐标向量，分别为网格行和列。这些向量的大小分别为 ny 和 nx
    y, x = torch.arange(ny), torch.arange(nx)
    # 使用 torch.meshgrid 生成二维网格坐标。输出 yv 和 xv 分别表示每个网格点的 y 坐标和 x 坐标。
    yv, xv = torch.meshgrid(y, x)
    # 堆叠 xv 和 yv，形成一个三维张量，其中最后一个维度（大小为2）代表每个网格点的 (x, y) 坐标。
    # - 0.5: 通过减去0.5给网格添加偏移量，使得网格中心落在每个网格单元的正中心。
    grid = torch.stack((xv, yv), 2)
    grid = grid.expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    anchor_grid = rgbt_anchors[i] * torch.tensor(stride[i])
    anchor_grid=anchor_grid.view(1, na, 1, 1, 2).expand(shape)
    #  每个网格点的 (x, y) 坐标。
    # anchor_grid: 处理后的锚框信息，适用于与推理输出计算。
    return grid, anchor_grid

# [[10,13], 
# [16,30],
# [33,23]]
def rgbt_postproces(input, nc,conf_thres=0.25, iou_thres=0.45,maxdets=100,maxbboxs=1024):

    no=5+nc
    na=3
    nl=3
    z=[]
    stride=[8., 16., 32.]
    grid= [torch.empty(0) for _ in range(nl)]
    anchor_grid = [torch.empty(0) for _ in range(nl)]

    for i in range(nl):
        input[i]=torch.from_numpy(input[i])
        bs, _, ny, nx = input[i].shape
        # x(bs,33,80,80) to x(bs,3,11,80,80) to x(bs,3,80,80,11)
        input[i]=input[i].view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        # input[i]=input[i].reshape([3, -1]+list(input[i].shape[-2:])) #(3,11,80,80)
        # input[i] = np.expand_dims(input[i], axis=0) 
        if grid[i].shape[2:4] != input[i].shape[2:4]:
            grid[i], anchor_grid[i] = _make_grid(na,stride,nx, ny, i)

        xy, wh, conf = sigmoid(input[i]).split((2, 2, nc + 1), 4)
        # 将 xy 转换为原始图像坐标。
        xy = (xy * 2 + grid[i]) * stride[i]  # xy
        # xy = xy.reshape(-1, 2)
        wh = (wh * 2) ** 2 * anchor_grid[i]  # wh
        y = torch.cat((xy, wh, conf), 4)
        # (1,3*20*20,11)
        z.append(y.view(bs, na * nx * ny, no))
        
    pred=(torch.cat(z, 1),)
    print("Data type of pred:", type(pred))
    pred = non_max_suppression(pred[0], conf_thres, iou_thres,maxdets,maxbboxs)[0].numpy()

    boxes = pred[:, :4]
    classes = pred[:, -1]
    scores = pred[:, 4]

    return boxes, classes, scores

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape ( (height, width))
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def rgbt_draw(original_img_rgb,original_img_t, resized_img_shape, original_img_shape, boxes, scores, classes,CLASSES):
    
    scaled_boxes=scale_boxes(resized_img_shape,boxes,original_img_shape)
    
    """    
        # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
    """

    for box, score, cl in zip(scaled_boxes, scores, classes):
        top, left, right, bottom = [round(_b) for _b in box]
        print('class: {}, score: {}'.format(CLASSES[round(cl)], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        if round(cl)==0:
            color=(255, 0, 0)
        elif round(cl)==1:
            color=(0, 0, 255)

        cv2.rectangle(original_img_rgb, (top, left), (right, bottom),color, 2)
        cv2.putText(original_img_rgb, '{0} {1:.2f}'.format(CLASSES[round(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
        cv2.rectangle(original_img_t, (top, left), (right, bottom), color, 2)
        cv2.putText(original_img_t, '{0} {1:.2f}'.format(CLASSES[round(cl)], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    
def pad_img(img):
    h = img.shape[0]
    w = img.shape[1]
    if h == w:
        pass
        return img, None
    elif h > w:
        img = cv2.copyMakeBorder(img, 0, 0, int(
            (h-w)/2), h-w-int((h-w)/2), cv2.BORDER_CONSTANT, 0)
        pad_par = [0, int((h-w)/2)]
    elif w > h:
        img = cv2.copyMakeBorder(
            img, int((w-h)/2), w-h-int((w-h)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
        pad_par = [int((w-h)/2), 0]
    return img, pad_par


def scale_img(img, imgz):
    assert img.shape[0] == img.shape[1], "不是矩形"
    gain = float(img.shape[0])/float(imgz)
    img = cv2.resize(img, (imgz, imgz))

    return img, gain


def getOrder(inputs):
    cont = list()
    for i in inputs:
        # 对于 inputs 中的每一个张量 i，获取其最后一个维度的大小（i.shape[-1]），并将其追加到 cont 列表中。
        cont.append(i.shape[-1])
    cont = np.array(cont)
    #排序前：cont=[80,40,20]
    return np.argsort(cont).tolist() # [3,2,1]


def check_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File Not Found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




def load_yaml(data):
    data = check_file(data)
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = int(data['nc'])
    CLASSES = data["names"]
    dataset = data["val"]
    annotations=data["annotations"]
    root_path=data["path"]
    def f(a): return map(lambda b: a[b:b+2], range(0, len(a), 2))
    ANCHOR = list()
    for i in data["anchors"]:
        ANCHOR.extend(list(f(i)))
    return root_path,nc, dataset,annotations,CLASSES, ANCHOR



def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

