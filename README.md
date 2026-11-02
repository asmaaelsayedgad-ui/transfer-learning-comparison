# Transfer Learning Comparison on CIFAR-10

A comparison of **ResNet50**, **MobileNetV2**, and **EfficientNetB0** as frozen ImageNet feature extractors for CIFAR-10 classification.

The project evaluates two different approaches on the same three backbones:

1. **Transfer learning** — freeze the pre-trained backbone and train a new classification head.
2. **Feature extraction + SVM** — extract pooled features once, then train a linear SVM classifier on top.

## Results

### Transfer learning (5 epochs, 10,000 test images)

| Model           | Test Loss | Test Accuracy |
| --------------- | --------: | ------------: |
| EfficientNetB0  |    0.2929 |    **90.21%** |
| ResNet50        |    0.3164 |        89.66% |
| MobileNetV2     |    0.4669 |        84.46% |

### Linear SVM on extracted features (10,000 test images)

| Feature Extractor | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-score |
| ----------------- | ------------------: | ------------: | -------------: | ----------: | ------------: |
| EfficientNetB0    |              89.08% |    **88.06%** |         0.8806 |      0.8806 |        0.8804 |
| ResNet50          |              87.62% |        87.74% |         0.8775 |      0.8775 |        0.8774 |
| MobileNetV2       |              85.12% |        84.58% |         0.8457 |      0.8457 |        0.8456 |

EfficientNetB0 achieves the best accuracy in both setups, while MobileNetV2 is the smallest and fastest backbone.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is loaded directly through `tf.keras.datasets.cifar10`:

- 45,000 training images
- 5,000 validation images (last 5,000 of the training split)
- 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Images are resized from `32x32` to `224x224` and preprocessed per backbone
(ResNet50 and MobileNetV2 use their own `preprocess_input`; EfficientNetB0 includes rescaling internally).

## Project structure

```
.
├── download_dataset.py        # Load CIFAR-10 and preview the pretrained backbones
├── transfer_learning.py       # Train a classification head on each frozen backbone
├── extract_features.py        # Extract pooled features (train/val/test) for each backbone
├── svm_feature_comparison.py  # Train and evaluate a LinearSVC on the extracted features
├── compare_models.py          # Quick backbone sanity check on 100 test images
├── dataset/                   # 100-image samples (CIFAR-10 and Open Images)
├── features/                  # Generated feature files (not tracked)
├── svm_results/
│   └── svm_comparison.csv     # SVM metrics for all backbones
├── transfer_learning_results.csv
├── model_comparison.csv
└── model_predictions.csv
```

Generated artifacts such as model weights (`*.keras`), feature files (`features/*.npz`), and the virtual environment are excluded via `.gitignore`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow scikit-learn pandas matplotlib numpy
```

## Usage

Run the scripts in order:

```bash
# 1. Inspect the dataset and pretrained backbones (optional)
python download_dataset.py

# 2. Transfer learning: train a new head on each frozen backbone
python transfer_learning.py

# 3. Extract features with each backbone
python extract_features.py

# 4. Train and compare LinearSVC classifiers on the features
python svm_feature_comparison.py

# 5. Quick check of each backbone on 100 test images
python compare_models.py
```

`transfer_learning.py` saves one model per backbone (`<Model>_cifar10.keras`), and
`extract_features.py` writes `features/<Model>_<split>_features.npz` files. Both are ignored by git,
so they are regenerated locally when the scripts run.

> **Note:** `compare_models.py` attaches a newly initialized (untrained) classification head to each
> pretrained backbone, so it is only a sanity check of the feature extractors and does not reflect the
> accuracy of the trained models above.

## Notes

- Training uses Adam (`lr=0.001`), sparse categorical cross-entropy, batch size 32, and 5 epochs.
- Only the classification head is trained; all pre-trained backbone layers stay frozen.
- The SVM pipeline uses `StandardScaler` followed by `LinearSVC(C=1.0, max_iter=5000)`.
