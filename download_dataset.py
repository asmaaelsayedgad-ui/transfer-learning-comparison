import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Download and load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", x_test.shape)
print("Testing labels:", y_test.shape)

# 2. Define the 10 classes
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 3. Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 4. Display some sample images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i][0])])
    plt.axis("off")

plt.tight_layout()
plt.show()

# 5. Load three pre-trained models with ImageNet weights
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0

models_dict = {
    "ResNet50": ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    ),
    "MobileNetV2": MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    ),
    "EfficientNetB0": EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
}

# Freeze the pre-trained model layers
for model in models_dict.values():
    model.trainable = False

print("\nThree pre-trained models loaded successfully!")

for name, model in models_dict.items():
    print(name, "output shape:", model.output_shape)