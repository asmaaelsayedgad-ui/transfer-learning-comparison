import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras import layers, Model

# 1. Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Same split used in the transfer-learning step
x_val = x_train[-5000:]
y_val = y_train[-5000:]

x_train = x_train[:-5000]
y_train = y_train[:-5000]

# 2. Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs("features", exist_ok=True)

# 3. Define the three pretrained models
model_functions = {
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
}


def prepare_dataset(images, model_name):
    images = tf.cast(images, tf.float32)

    if model_name == "ResNet50":
        images = resnet_preprocess(images)
    elif model_name == "MobileNetV2":
        images = mobilenet_preprocess(images)

    ds = tf.data.Dataset.from_tensor_slices(images)

    def resize_image(image):
        return tf.image.resize(image, IMAGE_SIZE)

    ds = ds.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# 4. Extract features using each model
for model_name, model_function in model_functions.items():
    print(f"\nExtracting features using {model_name}...")

    base_model = model_function(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    feature_maps = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(feature_maps)
    feature_extractor = Model(inputs, features)

    for split_name, images, labels in [
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        dataset = prepare_dataset(images, model_name)
        extracted_features = feature_extractor.predict(dataset, verbose=1)

        output_path = os.path.join("features", f"{model_name}_{split_name}_features.npz")
        np.savez_compressed(output_path, X=extracted_features, y=labels.flatten())

        print(f"{split_name} features shape:", extracted_features.shape)
        print("Saved:", output_path)

    del feature_extractor, base_model
    tf.keras.backend.clear_session()

print("\nFeature extraction completed!")
