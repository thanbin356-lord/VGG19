import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Đường dẫn đến tập dữ liệu
dataset_path = "E:/ModelLab/dataset/rps-cv-images"

def load_data(path):
    data = []
    labels = []
    class_names = os.listdir(path)
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for folder in class_names:
        sub_path = os.path.join(path, folder)
        for img in os.listdir(sub_path):
            image_path = os.path.join(sub_path, img)
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            data.append(img_arr)
            labels.append(class_indices[folder])
    
    return np.array(data) / 255.0, np.array(labels), class_names

# Load toàn bộ dữ liệu
data, labels, class_names = load_data(dataset_path)

# Chia dữ liệu thành train (80%), val (10%), test (10%)
train_x, temp_x, train_y, temp_y = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, stratify=temp_y, random_state=42)

# Lưu dữ liệu đã xử lý
np.savez_compressed("processed_data.npz", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, val_x=val_x, val_y=val_y, class_names=np.array(class_names))

print("Data preprocessing completed and saved!")
