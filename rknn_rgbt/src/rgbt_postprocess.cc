#include "rgbt_postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>

#include <set>
#include <vector>
#include <string>

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }


// 计算框的交集区域
inline float intersectionArea(const bbox& box1, const bbox& box2){
    float x1 = std::max(box1.x1, box2.x1);
    float y1 = std::max(box1.y1, box2.y1);
    float x2 = std::min(box1.x2, box2.x2);
    float y2 = std::min(box1.y2, box2.y2);
    if(x1 < x2 && y1 < y2) return (x2 - x1) * (y2 - y1);
    else return 0.0f;
};

// 计算矩形框的面积
inline float boxArea(const bbox& box){
  if(box.x1 < box.x2 && box.y1 < box.y2) 
    return (box.x2-box.x1) * (box.y2-box.y1);
    else
    printf("boxArea: error");
};

//多类别NMS
inline void NMSBoxes(std::vector<bbox>& input_boxes, const float nms_threshold, std::vector<bbox>& nmsed_boxes)
{
    std::vector<bbox> sortedBoxes = input_boxes;
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), [](bbox a, bbox b) {return a.confidence > b.confidence; });
    
    while(!sortedBoxes.empty()){
        
        // 每次取置信度最高的合法锚框，放入nms_boxes
        const bbox currentBox = sortedBoxes.front();
        nmsed_boxes.push_back(currentBox);
        sortedBoxes.erase(sortedBoxes.begin()); // 取完后从候选锚框中删除

        // 计算剩余锚框与置信度最高的锚框的IOU
        std::vector<bbox>::iterator it = sortedBoxes.begin();
        while (it != sortedBoxes.end()){
            const bbox candidateBox = *it; // 取当前候选锚框
            if (currentBox.cls_id==candidateBox.cls_id)
            {
                float intersection = intersectionArea(currentBox, candidateBox); // 计算候选框和合法框的交集面积
                float iou = intersection / (boxArea(currentBox) + boxArea(candidateBox) - intersection); // 计算iou
                if (iou >= nms_threshold) 
                {
                  sortedBoxes.erase(it);
                }  // 根据阈值过滤锚框，过滤完it指向下一个锚框
                else it++; // 保留当前锚框，判断下一个锚框
            }
            else
            {
                it++; // 保留当前锚框，判断下一个锚框
            }
        }
    }

    printf("input_boxes=%d, nmsed_boxes=%d\n", input_boxes.size(), nmsed_boxes.size());
};

static void clip_boxes(std::vector<bbox>& boxes, const std::vector<int>& img_shape) {
    // # Clip boxes (xyxy) to image shape (height, width)
    for (auto& box : boxes) {
        box.x1 = std::max(0.0f, std::min(box.x1, static_cast<float>(img_shape[1] - 1)));
        box.y1 = std::max(0.0f, std::min(box.y1, static_cast<float>(img_shape[0] - 1)));
        box.x2 = std::max(0.0f, std::min(box.x2, static_cast<float>(img_shape[1] - 1)));
        box.y2 = std::max(0.0f, std::min(box.y2, static_cast<float>(img_shape[0] - 1)));
    }
}

static std::vector<bbox> scale_boxes(const std::vector<bbox>& boxes, const std::vector<int>& resized_shape, const std::vector<int>&  original_shape) {
    // Rescale boxes (xyxy) from resized_shape to original_shape of the img ( (height, width))
    float gain = 0;
    std::pair<int, int> pad(0, 0);

     // calculate from img0_shape
    gain = std::min(static_cast<float>(resized_shape[0]) / original_shape[0], static_cast<float>(resized_shape[1]) / original_shape[1]);  // gain  = old / new
    pad = std::make_pair(static_cast<int>((resized_shape[1] - original_shape[1] * gain) / 2), static_cast<int>((resized_shape[0] - original_shape[0] * gain) / 2));  // wh padding
    
    std::vector<bbox> scaled_boxes = boxes;  // 复制原boxes

    for (auto& box : scaled_boxes) {
        box.x1 -= pad.first;  // x padding w
        box.x2 -= pad.first;
        box.y1 -= pad.second;  // y padding h
        box.y2 -= pad.second;

        box.x1 /= gain;
        box.x2 /= gain;
        box.y1 /= gain;
        box.y2 /= gain;
    }

    clip_boxes(scaled_boxes, original_shape);
    return scaled_boxes;
}

 void rgbt_postprocess(int8_t *input0, int8_t *input1, int8_t *input2, 
                        std::vector<bbox> detected_bobes, 
                        const std::vector<int>& resized_shape, const std::vector<int>&  original_shape, 
                        std::vector<int32_t> &qnt_zps,std::vector<float> &qnt_scales,
                        float conf_threshold, float nms_threshold)
{
  std::vector<bbox> input_boxes;
  std::vector<bbox> nmsed_boxes;
  // std::vector<bbox> scaled_boxes;
  

  //检测头坐标变换与合并


  //s1: 初步筛选掉多余的框
  // int validCount = 0;
  int grid_len=8400; //3 * 8400 = 25200
  for (int a = 0; a < 3; a++)
  {
    for (int j=0;j<8400;j++)
    {
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + j];
        if (box_confidence >= conf_threshold)
        {
          //保存当前检测框的xywh
          int offset = (PROP_BOX_SIZE * a) * grid_len + j;
          int8_t *in_ptr = input + offset;
          float box_x = *in_ptr;
          float box_y = in_ptr[1 * grid_len];
          float box_w = in_ptr[2 * grid_len];
          float box_h = in_ptr[3 * grid_len];
          
          // // 转换 xywh 到 xyxy
          float box_x1 = box_x;                       // 左上角 x
          float box_y1 = box_y;                       // 左上角 y
          float box_x2 = box_x + box_w;               // 右下角 x
          float box_y2 = box_y + box_h;               // 右下角 y          

          //计算类别置信度并保存对应的索引
          int8_t maxClassProbs = in_ptr[5 * grid_len] * box_confidence;
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k)
          {
            int8_t prob = in_ptr[(5 + k) * grid_len] * box_confidence;
            if (prob > maxClassProbs)
            {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
            bbox newBox;
            newBox.x1 = box_x1;
            newBox.y1 = box_y1;
            newBox.x2 = box_x2;
            newBox.y2 = box_y2;
            newBox.confidence = maxClassProbs; // 设置置信度
            newBox.cls_id = maxClassId; // 设置类别 ID

            // 将新的框添加到 input_boxes
            input_boxes.push_back(newBox);
            // validCount++;
        }
      }
    }




    //s2: 对筛选后的框进行多类别NMS
    NMSBoxes(input_boxes,nms_threshold,nmsed_boxes);
    // nmsed_boxes[(left,top,width,height,confidence,cls_id),......,]
    //s2: 将基于输入图像的框尺寸缩放回原图
    detected_bobes = scale_boxes(nmsed_boxes,resized_shape,original_shape);
}
