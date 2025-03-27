import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Load mô hình và danh sách lớp
model = load_model("rps_vgg19_model.h5")
class_names = np.load("class_names.npy", allow_pickle=True).tolist()

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx] * 100
    
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Predicted: {class_names[class_idx]}\nConfidence: {confidence:.2f}%")
    plt.show()

# Mở hộp thoại chọn file
Tk().withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
if file_path:
    predict_image(file_path)
