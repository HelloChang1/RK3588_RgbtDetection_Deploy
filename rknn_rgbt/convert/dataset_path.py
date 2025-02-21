import os

def save_image_paths_to_txt(val_folder, output_txt_file):
    # 创建一个空列表来存储路径
    image_pairs = []

    # 遍历 val 文件夹
    for filename in os.listdir(val_folder):
        if filename.endswith('_rgb.png'):
            # 获取对应的红外图像
            base_name = filename[:-8]  # 去掉 '_rgb.png'
            infrared_image_name = base_name + '_t.png'

            # 生成完整路径
            rgb_image_path = os.path.join(val_folder, filename)
            infrared_image_path = os.path.join(val_folder, infrared_image_name)

            # 检查红外图像是否存在
            if os.path.exists(infrared_image_path):
                # 将路径对添加到列表中
                image_pairs.append(f"{rgb_image_path} {infrared_image_path}")

    # 将路径对写入到 TXT 文件
    with open(output_txt_file, 'w') as f:
        for pair in image_pairs:
            f.write(pair + '\n')

    print(f"Image paths saved to {output_txt_file}")

# 示例用法
val_folder_path = './val/images/'  # 替换为 val 文件夹的实际路径
output_file_path = 'datasets.txt'   # 输出 TXT 文件名
save_image_paths_to_txt(val_folder_path, output_file_path)
