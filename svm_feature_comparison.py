import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Define the three feature extractors
model_names = [
    "ResNet50",
    "MobileNetV2",
    "EfficientNetB0"
]

os.makedirs("svm_results", exist_ok=True)

results = []

# 2. Train and test one SVM for each feature extractor
for model_name in model_names:

    print(f"\n{'=' * 50}")
    print(f"SVM using {model_name} features")
    print(f"{'=' * 50}")

    # Load training features
    train_data = np.load(
        f"features/{model_name}_train_features.npz"
    )
    X_train = train_data["X"]
    y_train = train_data["y"]

    # Load validation features
    val_data = np.load(
        f"features/{model_name}_val_features.npz"
    )
    X_val = val_data["X"]
    y_val = val_data["y"]

    # Load test features
    test_data = np.load(
        f"features/{model_name}_test_features.npz"
    )
    X_test = test_data["X"]
    y_test = test_data["y"]

    print("Training features:", X_train.shape)
    print("Validation features:", X_val.shape)
    print("Testing features:", X_test.shape)

    # 3. Create the SVM pipeline
    svm_model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        )
    )

    # 4. Train the SVM using training features
    print("\nTraining SVM...")
    svm_model.fit(X_train, y_train)

    # 5. Evaluate on the validation set
    val_predictions = svm_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # 6. Test the trained SVM on unseen test features
    print("\nTesting SVM...")
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")

    # 7. Display detailed classification results
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(10),
            target_names=[
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ],
            zero_division=0
        )
    )

    # 8. Plot the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "airplane", "auto", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        xticks_rotation="vertical",
        cmap="Blues"
    )

    plt.title(f"SVM Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    # 9. Save the results
    results.append({
        "Feature Extractor": model_name,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1-score": f1
    })

    # Release data before processing the next model
    del svm_model
    del X_train, X_val, X_test
    del y_train, y_val, y_test, y_pred

# 10. Compare the three SVM models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test Accuracy",
    ascending=False
)

print("\nFINAL SVM COMPARISON")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    "svm_results/svm_comparison.csv",
    index=False
)

# 11. Plot test accuracy comparison
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Feature Extractor"],
    results_df["Test Accuracy"]
)

plt.title("SVM Test Accuracy Using Different Features")
plt.xlabel("Pre-trained Feature Extractor")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

print("\nResults saved to svm_results/svm_comparison.csv")