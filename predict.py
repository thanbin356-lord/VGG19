import numpy as np
import cv2
import easygui
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load mô hình
model = load_model("E:/ModelLab/vgg19_rps_model.h5")

# Nhãn lớp
class_names = ["Paper", "Rock", "Scissors"]
threshold = 0.7  # Ngưỡng tin cậy (70%)

# Chọn ảnh bằng hộp thoại
img_path = easygui.fileopenbox(title="Chọn một ảnh", filetypes=["*.png", "*.jpg", "*.jpeg"])

if not img_path:  # Nếu không chọn ảnh
    print("❌ Không có ảnh nào được chọn!")
    exit()

# Xử lý ảnh
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

# Dự đoán
pred = model.predict(img_input)
confidence = np.max(pred)  # Độ tin cậy cao nhất
predicted_class = class_names[np.argmax(pred)]

# Kiểm tra ngưỡng tin cậy
if confidence < threshold:
    print(f"⚠️ Không chắc chắn (confidence = {confidence:.2f}). Bỏ qua dự đoán!")
else:
    print(f"✅ Ảnh dự đoán là: {predicted_class} (Độ tin cậy: {confidence:.2f})")

    # Hiển thị ảnh với kết quả dự đoán
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f})", fontsize=14, color="green")
    plt.show()
