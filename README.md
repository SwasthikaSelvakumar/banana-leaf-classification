# 🍌 Banana Leaf Disease Classification (BananaLSD)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SwasthikaSelvakumar/AI-Based-Waste-Detection-System-Using-CNN-and-Generative-AI-/blob/main/BananaLSD_Classifier_.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Task](https://img.shields.io/badge/Task-Image%20Classification-green)

A CNN-based classifier that identifies banana leaf diseases from a smartphone photo, built with **EfficientNetB0 transfer learning** and a **leakage-free evaluation protocol**.

| Test accuracy | Macro F1 | Test images | Training time |
|:---:|:---:|:---:|:---:|
| **0.99** (133/135) | **0.98** | 135 | ~6.2 min (Colab T4) |

---

## Problem

Banana growers lose yield when leaf-spot diseases are not caught early. This project classifies a leaf photo into one of four classes:

`Healthy` · `Sigatoka` · `Cordana` · `Pestalotiopsis`

The specific challenge is **data leakage**. The dataset ships with an *Augmented* set generated from the same photos as the *Original* set, so mixing the two would put near-copies of test images into training.

## Dataset

**BananaLSD** (Arman et al., 2023): 937 original 224×224 smartphone photos from banana fields in Bangladesh, labelled by a plant pathologist.

- Source: <https://data.mendeley.com/datasets/9tb7k297ff/1>
- **Only the `OriginalSet` is used.** The notebook never extracts or reads `AugmentedSet`.
- The dataset is **not included in this repo**. Download it from the link above.

| Class | Images (after duplicate removal) |
|---|:---:|
| Sigatoka | 473 |
| Pestalotiopsis | 172 |
| Cordana | 162 |
| Healthy | 91 |
| **Total** | **898** |

## How data leakage is prevented

1. **Original set only**, with the Augmented set excluded at extraction time.
2. **Exact-duplicate removal** by MD5 hash: 937 → 898 images (39 duplicates removed, 38 of them in the Healthy class, which would otherwise have leaked across splits).
3. **Stratified 70/15/15 split** (seed 42): 628 train / 135 validation / 135 test, made **before** any augmentation.
4. **Augmentation on the training stream only**, applied on the fly. Validation and test images are untouched.
5. The **test set is evaluated once**, after training.

## Method

| Component | Choice |
|---|---|
| Backbone | EfficientNetB0 (ImageNet weights), 4.05 M params |
| Head | GlobalAvgPool → BatchNorm → Dropout 0.4 → Dense 256 (ReLU, L2 1e-4) → Dropout 0.3 → Softmax(4) |
| Total params | 4.38 M |
| Input | 224×224×3 (EfficientNet's built-in preprocessing, 0–255 input) |
| Augmentation | Flips, rotation (±54°), zoom (±15%), translation (±10%), contrast, brightness |
| Imbalance handling | Class-weighted loss, stratified splits, macro-averaged metrics |
| Loss | Categorical cross-entropy, label smoothing 0.1 |
| Stage 1 | Backbone frozen, head trained. Adam, lr 1e-3, max 20 epochs |
| Stage 2 | Last 40 backbone layers unfrozen (BatchNorm kept frozen). Adam, lr 1e-5, max 40 epochs |
| Callbacks | Early stopping on val. accuracy (patience 8, best weights restored), ReduceLROnPlateau (×0.3) |
| Batch size | 32 |
| Inference | Test-time augmentation (original + horizontal flip + vertical flip) |

## Results (test set, n = 135)

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| Cordana | 0.96 | 0.96 | 0.96 | 25 |
| Healthy | 1.00 | 1.00 | 1.00 | 13 |
| Pestalotiopsis | 0.96 | 1.00 | 0.98 | 26 |
| Sigatoka | 1.00 | 0.99 | 0.99 | 71 |
| **Macro average** | **0.98** | **0.99** | **0.98** | 135 |

Test accuracy is 0.99 (133/135). With 135 test images one error is 0.74 points, and the 95% Wilson interval for accuracy is roughly 0.95–1.00.

<p align="center">
  <img src="images/learning_curves.png" width="62%" alt="Training and validation curves">
  <img src="images/confusion_matrix.png" width="30%" alt="Test confusion matrix">
</p>

### Error analysis

Only 2 of 135 test images were misclassified. Both involve a sparse lesion and had low confidence. No diseased leaf was labelled healthy.

<p align="center">
  <img src="images/error_analysis.png" width="90%" alt="Two misclassified and two low-confidence correct test images">
</p>

- **Sigatoka → Cordana (0.41):** only two or three tiny brown spots at the leaf edge against leaf litter (early-stage disease).
- **Cordana → Pestalotiopsis (0.35):** one isolated dark lesion with a yellow halo on an over-exposed leaf, resembling a Pestalotiopsis spot.
- All four images shown have confidence below 0.50, so a 0.50 confidence threshold would route both errors to expert review.

## Reproduce

1. Open the notebook in Google Colab (badge above) and set **Runtime → Change runtime type → T4 GPU**.
2. Download the BananaLSD zip from Mendeley Data.
3. Run all cells. When asked, upload the zip from your device. The notebook extracts only the Original set and finds the class folders automatically.

Environment used: Python 3 on Google Colab, TensorFlow/Keras 2.21, scikit-learn, pandas, matplotlib, seaborn.

## Repository structure

```
├── BananaLSD_Classifier_.ipynb   # full notebook with outputs
├── README.md
├── RegisterNo_PSxx_Report.pdf    # two-page report
└── images/
    ├── learning_curves.png
    ├── confusion_matrix.png
    └── error_analysis.png
```

## Limitations

- All photos come from one region, one month (June 2021) and three phones, so performance on other farms, seasons and cameras is untested.
- The model has no "unknown / other" class, so a different disease or nutrient deficiency is forced into one of four labels.
- The test set is small (135 images). There are no plant IDs, so near-duplicate shots of the same plant may span splits and the reported accuracy may be optimistic.
- Disease severity is not estimated.
- Predictions should be confidence-gated and confirmed by an expert before any treatment decision.

## Future work

Evaluate on photos from a different farm and season (domain shift), and add Grad-CAM to verify that the model attends to lesions rather than background.

## Citation

Dataset:
> Arman, S. E., Bhuiyan, M. A. B., Abdullah, H. M., Islam, S., Chowdhury, T. T., Hossain, M. A. (2023). *Banana Leaf Spot Diseases (BananaLSD) Dataset for Classification of Banana Leaf Diseases Using Machine Learning*. Mendeley Data, V1. doi:[10.17632/9tb7k297ff.1](https://doi.org/10.17632/9tb7k297ff.1)

Dataset paper: *BananaLSD: A banana leaf images dataset for classification of banana leaf diseases using machine learning*, Data in Brief (2023).

## Author

**Swasthika Selvakumar** · [GitHub](https://github.com/SwasthikaSelvakumar)

*AI-assistance declaration:* Claude (Anthropic) helped draft the notebook scaffolding and report text. The experiments were run and verified by the author.
