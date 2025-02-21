#ifndef _RGBT_POSTPROCESS_H_
#define _RGBT_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 6
#define CON_THRESH 0.25
#define IOU_THRESH 0.45
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

typedef struct _bbox 
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int cls_id;
} bbox;

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

void rgbt_postprocess(int8_t *input0, int8_t *input1, int8_t *input2,
                        std::vector<bbox> detected_bobes, 
                        const std::vector<int>& resized_shape, 
                        const std::vector<int>&  original_shape, 
                        std::vector<int32_t> &qnt_zps,
                        std::vector<float> &qnt_scales,
                        float conf_threshold, float nms_threshold);

#endif //_RGBT_POSTPROCESS_H_