import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, EfficientNetB0
)

# 1. Load CIFAR-10
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# Use a smaller validation set from the training data
x_val = x_train[-5000:]
y_val = y_train[-5000:]

x_train = x_train[:-5000]
y_train = y_train[:-5000]

# 2. Define the 10 classes
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 3. Resize images to 224x224 in batches
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 10

def prepare_dataset(images, labels, model_name, training=False):
    images = tf.cast(images, tf.float32)

    # Model-specific preprocessing
    if model_name == "ResNet50":
        images = tf.keras.applications.resnet50.preprocess_input(images)
    elif model_name == "MobileNetV2":
        images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    # EfficientNetB0 contains its own rescaling layer

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(5000)

    def resize_images(image, label):
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, tf.cast(tf.squeeze(label), tf.int32)

    ds = ds.map(
        resize_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds

# 4. Define the three pre-trained models
model_functions = {
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0
}

results = []

# 5. Train and evaluate each model
for model_name, model_function in model_functions.items():

    print(f"\n{'=' * 40}")
    print(f"Training {model_name}")
    print(f"{'=' * 40}")

    train_ds = prepare_dataset(
        x_train, y_train, model_name, training=True
    )
    val_ds = prepare_dataset(
        x_val, y_val, model_name
    )
    test_ds = prepare_dataset(
        x_test, y_test, model_name
    )

    # Load pre-trained feature extractor
    base_model = model_function(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pre-trained layers
    base_model.trainable = False

    # Add a new classifier for CIFAR-10
    inputs = layers.Input(shape=(224, 224, 3))
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(
        NUM_CLASSES, activation="softmax"
    )(features)

    model = Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the new classification head
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Evaluate on the complete CIFAR-10 test set
    test_loss, test_accuracy = model.evaluate(
        test_ds, verbose=1
    )

    print(f"{model_name} test accuracy: {test_accuracy:.4f}")

    results.append({
        "Model": model_name,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    })

    # Plot accuracy during training
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )
    plt.title(f"{model_name} Transfer Learning")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save trained model
    model.save(f"{model_name}_cifar10.keras")

    # Release memory before loading the next model
    del model, base_model, train_ds, val_ds, test_ds
    tf.keras.backend.clear_session()

# 6. Compare all three models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Test Accuracy", ascending=False
)

print("\nFINAL TRANSFER LEARNING RESULTS")
print(results_df.to_string(index=False))

results_df.to_csv("transfer_learning_results.csv", index=False)