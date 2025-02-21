from rknn.api import RKNN
import os
import cv2
import numpy as np
from utils.general import getOrder, easystream_post_process,easystream_draw,rgbt_postproces,rgbt_draw,TQDM_BAR_FORMAT,scale_boxes,process_batch,xywh2xyxy,LOGGER
from utils.metrics import ap_per_class
from utils.yolov5_post import *
import argparse
from utils.rgbt_dataset import create_rgbtdataloader
from tqdm import tqdm
import torch
from pathlib import Path

# conf_thres = 0.90
# iou_thres = 0.45
IMG_SIZE = (640, 640)
CLASSES = ("person", "car", "bus","motorcycle", "lamp", "truck")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rknn', type=str, default=None, help='Input  rknn model')
    parser.add_argument('--eval_acc', action='store_true', help='Evalate quantization accuracy of rknn model')

    opt = parser.parse_args()


    PLATFORM="rk3588"
    ONNX_PATH="rknn_rgbt/model/rgbt_ca_rtdetrv2_ours_add_original_op19_three_outputs_conv.onnx"
    RKNN_PATH="/rknn_rgbt/model/rgbt_ca_rtdetrv2_ours_add_original_op19_three_outputs_conv_fp16.rknn"
    val_path="rknn_rgbt/convert/val/images"
    # IMG_SIZE=640
    Acc_analysis=False
    Result_comparsion=False
    Easy_postprocess=True
    DO_QUANT=False
    Acc_eval=False
    SNAPSHOT_FP16_DIR="./SNAPSHOT_FP16"
    # DATASET='./datasets.txt'
    ANCHORS = [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0], 
              [30.0, 61.0], [62.0, 45.0], [59.0, 119.0], 
              [116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]
    # anchors=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
    RGBT_ANCHORS=[[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]
    # set inputs
    print('--> img_rgb and img_t load and resize')
    img_rgb = cv2.imread('rknn_rgbt/convert/val/images/05160004_rgb.png')
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb_plot=img_rgb
    img_rgb = letterbox(img_rgb, new_shape=(640, 640))[0]  # img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    resized_img_rgb=img_rgb
    img_rgb = np.expand_dims(img_rgb, axis=0)  # Shape will be (1, IMG_SIZE, IMG_SIZE, 3)
    img_t = cv2.imread('rknn_rgbt/convert/val/images/05160004_t.png')
    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
    img_t_plot=img_t
    # img_t = cv2.resize(img_t, (IMG_SIZE, IMG_SIZE))
    img_t = letterbox(img_t, new_shape=(640, 640))[0]  # img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    resized_img_t=img_t
    print(img_t.shape)
    img_t = np.expand_dims(img_t, axis=0)        # Shape will be (1, IMG_SIZE, IMG_SIZE, 3)
    if Acc_analysis:
        img_rgb = np.transpose(img_rgb, (0, 3, 1, 2))
        img_t = np.transpose(img_t, (0, 3, 1, 2))
    
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0,0,0],[0,0,0]],
                std_values=[[255,255,255],[255,255,255]],
                # std_values=[[1,1,1],[1,1,1]],
                target_platform=PLATFORM)
    print('done')

    if opt.rknn != None:
        ret=rknn.load_rknn(opt.rknn)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done') 
    else:  
        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(model=ONNX_PATH)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=DO_QUANT)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Accuracy analysis
        if Acc_analysis:
            print('--> Accuracy analysis')
            Ret = rknn.accuracy_analysis(inputs=[img_rgb,img_t],output_dir=SNAPSHOT_FP16_DIR)
            if ret != 0:
                print('Accuracy analysis failed!')
                exit(ret)
                print('done')

        # Export rknn model
        print('--> Export rknn model')
        ret = rknn.export_rknn(RKNN_PATH)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

    print('--> Init runtime environment')
    # init_runtime
    rknn.init_runtime();
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')


    if not Acc_analysis and not Acc_eval:
    
        print('--> Execute model inference')
        # Inference
        outputs = rknn.inference(inputs=[img_rgb,img_t]) 
        print("输出1,形状：{}".format(outputs[0].shape))
        print("输出2,形状：{}".format(outputs[1].shape))
        print("输出3,形状：{}".format(outputs[2].shape))
        Order = getOrder(outputs) # [3,2,1]
        print('Inference done')	
        if Result_comparsion:
            outputs[0]=outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:])) #(3,11,80,80)
            outputs[0] = np.expand_dims( outputs[0], axis=0) #(1,3,11,80,80)
            outputs[0]=np.transpose(outputs[0], (0, 1, 3, 4, 2))  #(1, 3,80,80, 11)
            outputs[0] = torch.tensor(outputs[0])  # 转换为张量
            outputs[0] = outputs[0].contiguous().view(1, -1, 11)  # 变换到 (1, 80*80*3, 11)
            print(outputs[0][0][0])
        
        print('--> postprocess model results ')
        # process
        img_rgb_plot = cv2.cvtColor(img_rgb_plot, cv2.COLOR_RGB2BGR)
        img_t_plot = cv2.cvtColor(img_t_plot, cv2.COLOR_RGB2BGR)
        
        if Easy_postprocess:
            # boxes, classes, scores = easystream_post_process(
            #     outputs, Order, ANCHORS, 640, 6, 0.25, 0.45)
            boxes, classes, scores = rgbt_postproces(outputs, 6)
            print('--> postprocess done')
            print('--> draw model results on img(512,640) ')
            rgbt_draw(img_rgb_plot,img_t_plot, resized_img_rgb.shape, img_rgb_plot.shape, boxes, scores, classes, CLASSES)
            cv2.imwrite("rknn_rgbt/convert/out_rgb_fp16_rgbt.jpg", img_rgb_plot)
            cv2.imwrite("rknn_rgbt/convert/out_t_fp16_rgbt.jpg", img_t_plot)
            print('draw done')

        else:
            boxes, classes, scores = yolov5_post_process(outputs) 
            # boxes, classes, scores = yolov5_post_process(
            #     outputs, Order, anchors, 640, 6, conf_thres, iou_thres)
            print('postprocess done')

            print('--> draw results to img')

            if boxes is not None:
                # draw(img_rgb_plot, boxes, scores, classes,CLASSES)
                # draw(img_t_plot, boxes, scores, classes,CLASSES)
                draw(img_rgb_plot, boxes, scores, classes)
                draw(img_t_plot, boxes, scores, classes)
            # show output or save output
            cv2.imwrite("out_rgb_fp16.jpg", img_rgb_plot)
            cv2.imwrite("out_t_fp16.jpg", img_t_plot)
            print('draw done')
    
    if Acc_eval:
        pad, rect = (0.5, False)
        dataloader = create_rgbtdataloader(val_path,
                                       640,
                                       1,
                                       32,
                                       False,
                                       pad=pad,
                                       rect=rect,
                                       workers=8)[0]    
        
        iouv = torch.linspace(0.5, 0.95, 10, device=None)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        jdict, stats, ap, ap_class = [], [], [], []
        classes_names={0: 'people', 1: 'car', 2: 'bus', 3: 'motorcycle', 4: 'lamp', 5: 'truck'}
        nc=6
        seen = 0
        print('-->on_val_start')
        pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for batch_i, (im_rgb, im_t, targets, rgb_paths, t_paths, shapes) in enumerate(pbar):
            seen += 1
            # print('-->on_val_batch_start')
            im_rgb=im_rgb.permute(0,2,3,1).numpy()
            im_t=im_t.permute(0,2,3,1).numpy()
            outputs = rknn.inference(inputs=[im_rgb,im_t]) 
            t_nb, t_height, t_width,_ = im_t.shape  # batch size, channels, height, width
            targets[:, 2:] *= torch.tensor((t_width, t_height, t_width, t_height),device=None)  # to pixels
            boxes, classes, scores = rgbt_postproces(outputs, 6)
            boxes=torch.from_numpy(boxes)
            classes=torch.from_numpy(classes)
            scores=torch.from_numpy(scores)
            # preds = [torch.zeros((0, 6), device=None)] * 1

            # for box,sco,cls in zip(boxes,scores,classes):
            #     tensor_pred = torch.cat((boxes, scores.unsqueeze(1), classes.unsqueeze(1)), dim=1)
            #     preds.append(tensor_pred)
            
            # for si, pred in enumerate(preds):
            #     labels = targets[targets[:, 0] == si, 1:]
            #     nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            #     path, shape = Path(t_paths[si]), shapes[si][0]
            #     predn = pred.clone()
            #     scale_boxes(im_t[si].shape[0:2], predn[:, :4], shape)  # native-space pred
            # for box,sco,cls in zip(boxes,scores,classes):
            labels = targets[:, 1:]
            nl, npr = labels.shape[0], boxes.shape[0]
            path= Path(t_paths[0])
            ori_shape=[512,640]
            res_shape=[640, 640]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=None)  # init

            boxesn=boxes.clone()
            scale_boxes(res_shape, boxesn, ori_shape)  # native-space pred
                        # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(res_shape, tbox, ori_shape)  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(boxesn,classes,labelsn, iouv)
            stats.append((correct, scores, classes, labels[:, 0]))  # (correct, conf, pcls, tcls)

        print('-->on_val_end')
        
        print('-->Compute metrics')
        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, names=classes_names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        print('Compute done')

        # Print results
        print(s)
        # pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        # LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        if nt.sum() == 0:
            print('WARNING no labels found in val set, can not compute metrics without labels')

        # Print results per class
        if (nc < 50 and nc > 1) and len(stats):
            for i, c in enumerate(ap_class):
                # LOGGER.info(pf % (classes_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                print(pf % (classes_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    # Release
    rknn.release()
