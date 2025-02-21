/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <dirent.h>
#define _BASETSD_H


#include "RgaUtils.h"

#include "rgbt_postprocess.h"

#include "rknn_api.h"

#include "preprocess.h"

#define PERF_WITH_POST 0

// namespace fs = std::experimental::filesystem;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
   if (argc < 3)
  {
    printf("Usage: %s <rknn model> <input_image_path> <resize/letterbox> <output_image_path>\n", argv[0]);
    return -1;
  }
  int ret;
  int ret1;
  int ret2;
  rknn_context ctx;
  size_t actual_size = 0;
  int img_width = 0;
  int img_height = 0;
  int img_channel = 0;
  const float conf_thres = CON_THRESH; // 默认的置信度阈值
  const float iou_thres = IOU_THRESH;      // 默认的NMS阈值
  struct timeval start_time, stop_time;
  char *model_name = (char *)argv[1];
  char *input_path = argv[2];
  std::string option = "resize";
  std::string out_path_rgb = "./img_rgb.jpg";
  std::string out_path_t = "./img_t.jpg";


  // init rga context
  rga_buffer_t src;
  rga_buffer_t dst;
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));

  /* Create the neural network */
  // 加载文件
  printf("Loading mode...\n");
  int model_data_size = 0;
  unsigned char *model_data = load_model(model_name, &model_data_size);
  ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  // 获取&打印版本信息
  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
  
  // 获取&打印输入输出数量
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
  
  // 获取&打印模型输入信息
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }
  
  // 获取&打印模型输入信息
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 3;
  int width = 0;
  int height = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    printf("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height = input_attrs[0].dims[2];
    width = input_attrs[0].dims[3];
  }
  else
  {
    printf("model is NHWC input fmt\n");
    height = input_attrs[0].dims[1];
    width = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }

  printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  rknn_input inputs[2];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  // inputs[0].size = width * height * channel;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;
  
  inputs[1].index = 1;
  inputs[1].type = RKNN_TENSOR_UINT8;
  // inputs[1].size = width * height * channel;
  inputs[1].fmt = RKNN_TENSOR_NHWC;
  inputs[1].pass_through = 0;

  //读图以及输入数据预处理

  std::string directory = input_path;
  std::string rgb_image_path;
  std::string t_image_path;

// 打开并遍历指定的目录
  DIR *dir = opendir(directory.c_str());
  if (dir == nullptr) {
      std::cerr << "Error: Could not open directory: " << directory << std::endl;
      return 1;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
      const std::string filename = entry->d_name;

      // 忽略 "." 和 ".."
      if (filename == "." || filename == "..") {
          continue;
      }

      // 根据后缀选择文件
      if (filename.find("_rgb") != std::string::npos &&
          filename.size() >= 4 && 
          filename.compare(filename.size() - 4, 4, ".png") == 0) {
          rgb_image_path = directory + "/" + filename; // 拼接完整路径
      } 
      else if (filename.find("_t") != std::string::npos &&          
                filename.size() >= 4 && 
                filename.compare(filename.size() - 4, 4, ".png") == 0) {
          t_image_path = directory + "/" + filename; // 拼接完整路径
      }
  }

  closedir(dir); // 关闭目录流
  
  // 检查文件路径是否有效
  if (rgb_image_path.empty()) {
      std::cout << "Error: No file found for RGB image (with suffix _rgb)." << std::endl;
      return -1;
  }

  if (t_image_path.empty()) {
      std::cout << "Error: No file found for Thermal image (with _t suffix)." << std::endl;
      return -1;
  }

  // 读取 RGB 图像
  cv::Mat orig_img_rgb = cv::imread(rgb_image_path, cv::IMREAD_COLOR);
  if (orig_img_rgb.empty()) {
      std::cout << "Error: Unable to read RGB image: " << rgb_image_path << std::endl;
      return -1;
  }

  // 读取 T 图像
  cv::Mat orig_img_t = cv::imread(t_image_path, cv::IMREAD_COLOR);
  if (orig_img_t.empty()) {
      std::cout << "Error: Unable to read T image: " << t_image_path << std::endl;
      return -1;
  }
  // ......
  cv::Mat img_rgb;
  cv::Mat img_t;
  cv::cvtColor(orig_img_rgb, img_rgb, cv::COLOR_BGR2RGB);
  cv::cvtColor(orig_img_t, img_t, cv::COLOR_BGR2RGB);

  img_width = img_rgb.cols;
  img_height = img_rgb.rows;
  printf(" original img width = %d, img height = %d\n", img_width, img_height);
  

  // 指定目标大小和预处理方式,默认使用LetterBox的预处理
  cv::Size target_size(width, height);
  cv::Mat resized_img_rgb(target_size.height, target_size.width, CV_8UC3);  
  cv::Mat resized_img_t(target_size.height, target_size.width, CV_8UC3);  
  // 计算缩放比例
  float scale_w = (float)target_size.width / img_rgb.cols;
  float scale_h = (float)target_size.height / img_rgb.rows;
  
    if (img_width != width || img_height != height)
  {
    // 直接缩放采用RGA加速
    if (option == "resize")
    {
      printf("resize image by rga\n");
      ret1 = resize_rga(src, dst, img_rgb, resized_img_rgb, target_size);
      ret2 = resize_rga(src, dst, img_t, resized_img_t, target_size);
      if (ret1 != 0 || ret2 != 0)
      {
        fprintf(stderr, "resize with rga error\n");
        return -1;
      }
      // 保存预处理图片
      // cv::imwrite("resize_input.jpg", resized_img);

    }

    
    // inputs[0].size = resized_img_rgb.total() * static_cast<uint32_t>(resized_img_rgb.channels()) * 2;
    inputs[0].size = width * height * channel;
    inputs[0].buf = resized_img_rgb.data;
    inputs[1].size = width * height * channel;
    // inputs[1].size = static_cast<uint32_t>(resized_img_t.total()) *  static_cast<uint32_t>(resized_img_t.channels()) * 2;
    inputs[1].buf = resized_img_t.data;
    printf(" resized img width = %d, img height = %d, intput_size1: %lld\n", resized_img_rgb.cols, resized_img_rgb.rows,resized_img_rgb.total() * resized_img_rgb.channels());
  }

  //读图以及输入数据预处理
  
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, io_num.n_input, inputs);

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    outputs[i].index = i;
    outputs[i].want_float = 0;
  }

  // 执行推理
  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  // 后处理
  // nms,annotator
  printf("post process config: conf_thres = %.2f, iou_thres = %.2f\n", conf_thres, iou_thres);
  std::vector<bbox> detected_boxes;
  const std::vector<int> resized_shape={height,width};   //HW
  const std::vector<int>  original_shape={img_height,img_width}; // HW
  rgbt_postprocess((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, detected_boxes, resized_shape, original_shape, conf_thres, iou_thres);
  
  //画框
  char text[256];
  int count=0;
  std::vector<std::string> names = {
    "car",
    "person"
    "bus",          // Uncomment if needed
    "motorcycle",   // Uncomment if needed
    "lamp",         // Uncomment if needed
    "truck"         // Uncomment if needed
  };
  for(auto& box: detected_boxes)
  {
    count++;
    sprintf(text, "%s %.1f%%", names[box.cls_id], box.confidence * 100);
    rectangle(orig_img_rgb, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(256, 0, 0, 256), 3);
    rectangle(orig_img_t, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(256, 0, 0, 256), 3);
    putText(orig_img_rgb, text, cv::Point(box.x1, box.y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    putText(orig_img_t, text, cv::Point(box.x1, box.y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }
  printf("total box counts in img:%d\n",count);
  printf("save detect result to %s\n", out_path_rgb.c_str());
  printf("save detect result to %s\n", out_path_t.c_str());
  imwrite(out_path_rgb, orig_img_rgb);
  imwrite(out_path_t, orig_img_rgb);
  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

  // 耗时统计
  int loops = 10;
  int batch_size =1;
  double time_use_rknn=0.00;
  //开始计时
  gettimeofday(&start_time, NULL);

  for (int i = 0; i < loops; ++i)
  {
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
#if PERF_WITH_POST
  post_process((int8_t *)outputs[0].buf, detected_boxes, resized_shape, original_shape, box_conf_threshold, nms_threshold);
#endif
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
  }
  //结束计时
  gettimeofday(&stop_time, NULL);
  time_use_rknn = __get_us(stop_time) - __get_us(start_time); //ms
  printf("loop counts:%d , average rknn run:%f ms , rknn run FPS(10 loops):%f fps\n", loops,
          time_use_rknn/ 1000.0 / loops, loops * batch_size * 1000 / time_use_rknn);

  // release
  ret = rknn_destroy(ctx);

  if (model_data)
  {
    free(model_data);
  }

  return 0;
}