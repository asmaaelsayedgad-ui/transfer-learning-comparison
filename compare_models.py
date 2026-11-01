import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, EfficientNetB0
)
from download_dataset import x_test, y_test

# Select 100 images from the test dataset
x_sample = x_test[:100]
y_sample = y_test[:100].flatten()

# Resize images from 32x32 to 224x224
x_sample_resized = tf.image.resize(
    x_sample, (224, 224)
)

# Define the three pre-trained models
model_functions = {
    "ResNet50": ResNet50,
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0
}

results = []

for name, model_function in model_functions.items():

    print(f"\nTesting {name}...")

    # Load ImageNet pre-trained feature extractor
    base_model = model_function(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Add a CIFAR-10 classification head
    inputs = layers.Input(shape=(224, 224, 3))
    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    outputs = layers.Dense(10, activation="softmax")(features)

    model = Model(inputs, outputs)

    # Apply model-specific preprocessing
    if name == "ResNet50":
        x_processed = tf.keras.applications.resnet50.preprocess_input(
            x_sample_resized * 255.0
        )
    elif name == "MobileNetV2":
        x_processed = tf.keras.applications.mobilenet_v2.preprocess_input(
            x_sample_resized * 255.0
        )
    else:
        # EfficientNetB0 includes rescaling in its model
        x_processed = x_sample_resized

    # Predict the classes of the 100 images
    predictions = model.predict(
        x_processed, batch_size=10, verbose=0
    )

    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_sample) * 100

    results.append({
        "Model": name,
        "Images Tested": len(y_sample),
        "Correct Predictions": int(
            np.sum(predicted_classes == y_sample)
        ),
        "Accuracy (%)": round(accuracy, 2)
    })

    print(f"Accuracy: {accuracy:.2f}%")

    # Release model memory
    del model, base_model
    tf.keras.backend.clear_session()

# Display comparison
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON ON 100 IMAGES")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("model_test_results.csv", index=False)
print("\nResults saved to model_test_results.csv")