import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dữ liệu
data = np.load("processed_data.npz", allow_pickle=True)
train_x, train_y = data["train_x"], data["train_y"]
val_x, val_y = data["val_x"], data["val_y"]
class_names = data["class_names"].tolist()

# Data Augmentation (Nhẹ)
datagen = ImageDataGenerator(
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=True
)

# Load mô hình VGG19
vgg = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Chỉ đóng băng 10 layer đầu tiên để có thể fine-tune phần còn lại
for layer in vgg.layers[:10]:
    layer.trainable = False

# Thêm Fully Connected Layers
x = Flatten()(vgg.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout 50%
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)  # Dropout 30%
prediction = Dense(len(class_names), activation='softmax')(x)

# Xây dựng mô hình
model = Model(inputs=vgg.input, outputs=prediction)

# Biên dịch mô hình
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Early Stopping để tránh overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)

# Huấn luyện mô hình với Data Augmentation
history = model.fit(
    datagen.flow(train_x, train_y, batch_size=32),
    validation_data=(val_x, val_y),
    epochs=15,
    callbacks=[early_stop],
    shuffle=True
)

# Lưu mô hình và class names
model.save("rps_vgg19_improved.h5")
np.save("class_names.npy", class_names)

# Vẽ biểu đồ độ chính xác
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig('vgg-improved-acc-rps.png')
plt.show()

# Vẽ biểu đồ loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig('vgg-improved-loss-rps.png')
plt.show()

print("✅ Model training completed and saved!")
