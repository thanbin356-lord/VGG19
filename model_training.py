import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping

# Load dữ liệu
data = np.load("processed_data.npz", allow_pickle=True)
train_x, train_y = data["train_x"], data["train_y"]
val_x, val_y = data["val_x"], data["val_y"]
class_names = data["class_names"].tolist()

# Xây dựng mô hình VGG19
vgg = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

# Biên dịch mô hình
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Sử dụng Early Stopping để tránh overfitting
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Huấn luyện mô hình
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, callbacks=[early_stop], batch_size=32, shuffle=True)

# Lưu mô hình
model.save("rps_vgg19_model.h5")
np.save("class_names.npy", class_names)

# Vẽ biểu đồ độ chính xác
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('vgg-acc-rps.png')
plt.show()

# Vẽ biểu đồ loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('vgg-loss-rps.png')
plt.show()

print("Model training completed and saved!")