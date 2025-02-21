import argparse
import os
from tqdm import tqdm
import cv2
from rknn.api import RKNN
import pkg_resources

from utils.general import *
from utils.eval import *


version=pkg_resources.get_distribution("rknn-toolkit2").version[:3]
print(f"\033[0;37;42mUsing SDK:{version}\033[0m")

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('-i', '--input_onnx', required=True, type=str, help='Input onnx model')
    parser.add_argument('-o', '--output_rknn', type=str, default='rknn', help='Save rknn model path')
    parser.add_argument('--rknn', type=str, default=None, help='Input  rknn model')

    # Device
    parser.add_argument('--platform', type=str, default='rk3588', choices=['rk3588', 'rk3566'], help='Set target platform, e.g. rk3588, rk3566')

    # Inference
    parser.add_argument('--data', type=str, default='data/visdrone.yaml', help='*.data path')
    parser.add_argument('--imgz', type=int, default=640, help='Model input img size')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--maxdets', type=int, default=100, help='maximum number of detections per image')
    parser.add_argument('--maxbboxs', type=int, default=1024, help='maximum number of boxes into torchvision.ops.nms()')

    # Mode
    parser.add_argument('--qnt_algo', type=str, default='normal', choices=['normal', 'mmse','kl_divergence'], help='quantized_algorithm: currently support: normal, mmse (Min Mean Square Error),kl_divergence(only sdk1.4).')
    parser.add_argument('--qnt_meth', type=str, default='channel', choices=['layer', 'channel'], help='quantized_method: quantize method, currently support: layer, channel.')
    parser.add_argument('--build', action='store_true', help='Export rknn model (default = disabled)')
    # parser.add_argument('--buildOnly', action='store_true', help='Exit after the rknn model has been built and skip inference perf measurement (default = disabled)')
    parser.add_argument('--sim', action='store_true', help='Simulate execution on PC')
    parser.add_argument('--qnt', action='store_true', help='do_quantization or not')
    parser.add_argument('--eval_perf', action='store_true',help='Evalate performance of rknn model')
    parser.add_argument('--debug_perf', action='store_true', help='Debug performance of each layer in a rknn model')
    parser.add_argument('--eval_mem', action='store_true', help='Evalate memory of rknn model')
    parser.add_argument('--eval_acc', action='store_true', help='Evalate quantization accuracy of rknn model')
    parser.add_argument('--test', action='store_true', help='Inference all dataset')  
    
    if version == "1.4":
        parser.add_argument('--single_core_mode', type=bool,default=False, help='set single_core_mode, default False,only RK3588')

    # Eval
    parser.add_argument('--eval', action='store_true', help='Evaluation accuracy,Eval require save reslut txt')

    # Save
    parser.add_argument('--project', default=None, help='save to project/name ,defult :runs/(modelName+qnt)')
    parser.add_argument('--save_txt', action='store_true',help="Save result txt")
    parser.add_argument('--save_img', action='store_true',help="Save result img")

    opt = parser.parse_args()
    root_path, nc, dataset, annotations, CLASSES, ANCHOR = load_yaml(opt.data)
    # Create RKNN object
    rknn = RKNN(verbose=True)
    try:
        # pre-process config
        print('--> Config model')
        if version == "1.4":
            rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=opt.platform, quantized_algorithm=opt.qnt_algo, quantized_method=opt.qnt_meth,single_core_mode=opt.single_core_mode)
        elif version =="1.3":
            rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=opt.platform, quantized_algorithm=opt.qnt_algo, quantized_method=opt.qnt_meth)
        elif version=="2.3":
            rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=opt.platform, quantized_algorithm=opt.qnt_algo, quantized_method=opt.qnt_meth,quant_img_RGB2BGR=[True,True])
        print('done')

        # Load onnx model
        print('--> Loading model')
        if opt.rknn != None:
            ret=rknn.load_rknn(opt.rknn)
            if ret != 0:
                print('Load model failed!')
                exit(ret)
            print('done')
        else:
            ret = rknn.load_onnx(model=opt.input_onnx)
            if ret != 0:
                print('Load model failed!')
                exit(ret)
            print('done')

            # Build model
            print('--> Building model')
            ret = rknn.build(do_quantization=opt.qnt, dataset=dataset)
            if ret != 0:
                print('Build model failed!')
                exit(ret)
            print('done')

            if opt.build:
                # Export RKNN model
                check_path(opt.output_rknn)
                filename = opt.input_onnx.split("/")[-1].replace("onnx", "rknn")
                output_dir = os.path.join(opt.output_rknn, filename)
                print('--> Export rknn model')
                ret = rknn.export_rknn(output_dir)
                if ret != 0:
                    print('Export rknn model failed!')
                    exit(ret)
                print('done')

        # Init runtime environment
        print('--> Init runtime environment')
        if opt.sim and opt.rknn ==None:
            # TARTGET=None
            if version == "1.4":
                ret = rknn.init_runtime()
            elif version == "1.3":
                ret = rknn.init_runtime(eval_mem=opt.eval_mem, perf_debug=opt.debug_perf)
            elif version == "2.3":
                ret = rknn.init_runtime()
        else:
            # TARTGET =
            ret = rknn.init_runtime(target= opt.platform, eval_mem=opt.eval_mem, perf_debug=opt.debug_perf)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')

        # param init
        img = cv2.imread("./test.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (opt.imgz, opt.imgz))

        if not (version == "1.4" and opt.sim):
            if (opt.eval_perf or opt.debug_perf) :
                print('--> Runing performance evaluation')
                outputs = rknn.eval_perf(inputs=[img])
                print('done')
            elif opt.eval_mem:
                print('--> Runing memory evaluation')
                outputs = rknn.eval_memory()
                print('done')
            elif opt.eval_acc:
                print('--> Runing accuracy_analysis evaluation')
                outputs = rknn.accuracy_analysis(
                    inputs=[img], target=opt.platform, device_id=opt.device)
                print('done')
        if opt.test:
            print('--> output info')
            outputs = rknn.inference(inputs=[img])
            print("输出1,形状：{}".format(outputs[0].shape))
            print("输出2,形状：{}".format(outputs[1].shape))
            print("输出3,形状：{}".format(outputs[2].shape))
            Order = getOrder(outputs)
            # Inference
            print('--> Running model')
            with open(dataset, "r", encoding="utf-8") as f:
                datasets = f.readlines()

            modelName=opt.input_onnx.split("/")[-1].replace(".onnx","")
            if opt.project == None:
                if opt.qnt:
                    opt.project = "runs/{}_qnt".format(modelName)
                else:
                    opt.project = "runs/{}".format(modelName)

            check_path(os.path.join(opt.project, "labels"))
            check_path(os.path.join(opt.project, "images"))
            check_path(os.path.join(opt.project, "json"))
            # print(datasets)
            for index, img_path in enumerate(tqdm(datasets)):
                img0 = cv2.imread(img_path[:-1])
                img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                img1, pad_par = pad_img(img1)
                img1, gain = scale_img(img1, opt.imgz)
                if opt.sim and opt.rknn ==None:
                    img1 = img1.reshape(1, opt.imgz, opt.imgz, 3)
                outputs = rknn.inference(inputs=[img1])

                # post process
                boxes, classes, scores = yolov5_post_process(
                    outputs, Order, ANCHOR, opt.imgz, nc, opt.conf_thres, opt.iou_thres,opt.maxdets,opt.maxbboxs)

                if opt.save_txt:
                    txt_name = img_path[:-1].split("/")[-1][:-3]+"txt"
                    with open(os.path.join(opt.project, "labels", txt_name), "w", encoding="utf-8") as f:
                        boxes = scale_coords(pad_par, gain, boxes, img0.shape)
                        save_txt(img0, boxes, scores, classes, f)
                # show output
                if opt.save_img :
                    img_name = img_path[:-1].split("/")[-1]
                    draw(img0, boxes, scores, classes, CLASSES)
                    cv2.imwrite(os.path.join(opt.project,"images", img_name), img0)
            print("done")
            if opt.eval and opt.save_txt:
                print("--> Start eval....")
                ImgaeDir = os.path.join(root_path,"val", "images")
                LabelsDir = os.path.join(opt.project, "labels")
                pred_dict = yolo2coco(ImgaeDir, LabelsDir, CLASSES)
                folder = os.path.join(opt.project, "json")
                pred_json=save_json(folder, "dt.json", pred_dict)
                evalByCoCo(annotations, pred_json)
    finally:
        rknn.release()
