import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def load_yolo_labels(label_path):
    """
    ���ļ��м���YOLO��ǩ��

    :param label_path: ��ǩ�ļ�·��
    :return: YOLO��ǩ�б�ÿ����ǩΪ(class_id, center_x, center_y, width, height)
    """
    yolo_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            # ��ȡÿ�в�����Ϊ class_id, center_x, center_y, width, height
            values = list(map(float, line.strip().split()))
            yolo_labels.append(tuple(values))
    return yolo_labels

def plot_yolo_label(image_path, label_path):
    """
    ����YOLO��ǩ�ı߽��

    :param image_path: ����ͼ����ļ�·��
    :param label_path: YOLO��ǩ�ļ�·��
    """
    # ʹ��PIL��ͼ��
    image = Image.open(image_path)
    image = np.array(image)  # ת��Ϊnumpy����

    # ����YOLO��ǩ
    yolo_labels = load_yolo_labels(label_path)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    image_height, image_width, _ = image.shape

    for label in yolo_labels:
        class_id, center_x, center_y, bbox_width, bbox_height = label

        # YOLO�������ǹ�һ���ģ����ｫ����ת�������ص�λ
        bbox_width *= image_width
        bbox_height *= image_height
        center_x *= image_width
        center_y *= image_height

        # �������Ͻǵ�����
        x_min = center_x - (bbox_width / 2)
        y_min = center_y - (bbox_height / 2)

        # ����һ�����ο���ӵ�ͼ����
        rect = patches.Rectangle((x_min, y_min), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # �������ı�
        ax.text(x_min, y_min, f"Class {int(class_id)}", color='white', backgroundcolor='red', fontsize=12)

    plt.show()




if __name__ == "__main__":
    image_path = "/home/liuyvjie/data/polyps/imagesy/1.jpg"
    label_path = "/home/liuyvjie/data/polyps/labels/1.txt"

plot_yolo_label(image_path, label_path)
