import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def load_yolo_labels(label_path):
    """
    从文件中加载YOLO标签。

    :param label_path: 标签文件路径
    :return: YOLO标签列表，每个标签为(class_id, center_x, center_y, width, height)
    """
    yolo_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            # 读取每行并解析为 class_id, center_x, center_y, width, height
            values = list(map(float, line.strip().split()))
            yolo_labels.append(tuple(values))
    return yolo_labels

def plot_yolo_label(image_path, label_path):
    """
    绘制YOLO标签的边界框。

    :param image_path: 输入图像的文件路径
    :param label_path: YOLO标签文件路径
    """
    # 使用PIL打开图像
    image = Image.open(image_path)
    image = np.array(image)  # 转换为numpy数组

    # 加载YOLO标签
    yolo_labels = load_yolo_labels(label_path)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    image_height, image_width, _ = image.shape

    for label in yolo_labels:
        class_id, center_x, center_y, bbox_width, bbox_height = label

        # YOLO的坐标是归一化的，这里将它们转换回像素单位
        bbox_width *= image_width
        bbox_height *= image_height
        center_x *= image_width
        center_y *= image_height

        # 计算左上角的坐标
        x_min = center_x - (bbox_width / 2)
        y_min = center_y - (bbox_height / 2)

        # 创建一个矩形框并添加到图像中
        rect = patches.Rectangle((x_min, y_min), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 添加类别文本
        ax.text(x_min, y_min, f"Class {int(class_id)}", color='white', backgroundcolor='red', fontsize=12)

    plt.show()




if __name__ == "__main__":
    image_path = "/home/liuyvjie/data/polyps/imagesy/1.jpg"
label_path = "/home/liuyvjie/data/polyps/labels/1.txt"

plot_yolo_label(image_path, label_path)
