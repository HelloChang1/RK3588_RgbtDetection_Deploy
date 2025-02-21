#include "preprocess.h"

int letterbox(const cv::Mat &image, cv::Mat &new_image, BOX_RECT &pads, const cv::Scalar& color)
{
    cv::Size shape(image.cols, image.rows);  // current shape [width, height]
    cv::Size new_shape(new_image.cols, new_image.rows);  // new shape [width, height]

    float r = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);
   
    std::pair<float, float> ratio(r, r);  // width, height ratios
    cv::Size new_unpad(static_cast<int>(std::round(shape.width * r)), static_cast<int>(std::round(shape.height * r)));

    float dw = new_shape.width - new_unpad.width; //0
    float dh = new_shape.height - new_unpad.height;  // 64 wh padding

    dw /= 2.0f;  // divide padding into 2 sides
    dh /= 2.0f;

    pads.left = static_cast<int>(std::round(dw - 0.1f));//0
    pads.right =static_cast<int>(std::round(dw + 0.1f));
    pads.top =static_cast<int>(std::round(dh - 0.1f));//64
    pads.bottom = static_cast<int>(std::round(dh + 0.1f)); //64

    // 在图像周围添加填充
    cv::copyMakeBorder(image, new_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, color);
    return 0;
}

int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size)
{
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    size_t img_width = image.cols;
    size_t img_height = image.rows;
    if (image.type() != CV_8UC3)
    {
        printf("source image type is %d!\n", image.type());
        return -1;
    }
    size_t target_width = target_size.width;
    size_t target_height = target_size.height;
    src = wrapbuffer_virtualaddr((void *)image.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resized_image.data, target_width, target_height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        fprintf(stderr, "rga check error! %s", imStrError((IM_STATUS)ret));
        return -1;
    }
    IM_STATUS STATUS = imresize(src, dst);
    return 0;
}
