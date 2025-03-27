import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Load dữ liệu kiểm tra
data = np.load("processed_data.npz", allow_pickle=True)
test_x, test_y = data["test_x"], data["test_y"]
class_names = data["class_names"].tolist()

# Load mô hình
model = load_model("rps_vgg19_improved.h5")

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(test_x, test_y, batch_size=32)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Dự đoán
y_pred = model.predict(test_x)
y_pred = np.argmax(y_pred, axis=1)

# Báo cáo phân loại
print("Classification Report:")
print(classification_report(test_y, y_pred, target_names=class_names))

# Ma trận nhầm lẫn
print("Confusion Matrix:")
print(confusion_matrix(test_y, y_pred))

# Vẽ ma trận nhầm lẫn
cm = confusion_matrix(test_y, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

print("Model evaluation completed!")
