import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np

# Đường dẫn dataset
data_dir = "E:/ModelLab/dataset/rps-cv-images"

# Tiền xử lý ảnh
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load mô hình VGG19
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Đóng băng các layer của mô hình gốc
for layer in base_model.layers:
    layer.trainable = False

# Thêm các layer mới
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(3, activation="softmax")(x)  # 3 lớp: Kéo, Búa, Bao

model = Model(inputs=base_model.input, outputs=x)

# Biên dịch mô hình
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(train_generator, validation_data=val_generator, epochs=20)

# Lưu mô hình
model.save("E:/ModelLab/vgg19_rps_model.h5")

print("✅ Huấn luyện hoàn tất, mô hình đã được lưu.")
