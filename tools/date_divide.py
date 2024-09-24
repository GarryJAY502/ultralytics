import os
import random
import shutil

# 设置路径
image_dir = '/home/liuyvjie/data/polyps/imagesy'  # 图片文件夹
label_dir = '/home/liuyvjie/data/polyps/labels'  # 标签文件夹

# 外层目录结构
train_img_dir = '/home/liuyvjie//data/polyps/divide/imagesy/train/'
val_img_dir = '/home/liuyvjie//data/polyps/divide/imagesy/val/'
test_img_dir = '/home/liuyvjie//data/polyps/divide/imagesy/test/'

train_lbl_dir = '/home/liuyvjie//data/polyps/divide/labels/train/'
val_lbl_dir = '/home/liuyvjie//data/polyps/divide/labels/val/'
test_lbl_dir = '/home/liuyvjie//data/polyps/divide/labels/test/'

# 创建图片和标签的输出文件夹
for folder in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 随机打乱数据
random.shuffle(image_files)

# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 计算每个数据集的大小
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)
test_count = total_images - train_count - val_count

# 分配数据集
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]


def move_files(files, image_dir, label_dir, target_img_dir, target_lbl_dir):
    for file in files:
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 移动图片到对应的 images 文件夹
        shutil.copy(image_path, os.path.join(target_img_dir, file))

        # 移动标签文件到对应的 labels 文件夹
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(target_lbl_dir, os.path.basename(label_path)))


# 移动文件到相应的文件夹
move_files(train_files, image_dir, label_dir, train_img_dir, train_lbl_dir)
move_files(val_files, image_dir, label_dir, val_img_dir, val_lbl_dir)
move_files(test_files, image_dir, label_dir, test_img_dir, test_lbl_dir)

print(f"数据集划分完成：训练集 {len(train_files)}，验证集 {len(val_files)}，测试集 {len(test_files)}")
