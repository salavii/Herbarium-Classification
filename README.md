# 🌿 Herbarium Classification — Fine-Grained Plant Species Recognition

Deep learning pipeline for classifying digitized herbarium specimens across **683 plant species**, built with PyTorch. The project walks through a full diagnosis-and-fix cycle: a custom CNN baseline that overfits catastrophically, then a transfer-learning solution with augmentation and regularization that recovers **~5× the validation accuracy**.

Dataset: [Kaggle Herbarium 2019](https://www.kaggle.com/c/herbarium-2019-fgvc6)

---

## 📊 Results

| Model | Train Acc | Val Acc | Notes |
|---|---|---|---|
| Custom CNN (2-layer, SGD) | 99.97% | **15.04%** | Severe overfitting — memorized the training set |
| ResNet-50 + augmentation + dropout | 99.99% | **73.80%** | Best model (Adam, lr=1e-4, LR scheduling) |

**Final model metrics (weighted, 683 classes):**

| Metric | Score |
|---|---|
| Validation Accuracy | **73.80%** |
| F1-Score | 0.676 |
| Precision | 0.690 |
| Recall | 0.692 |

> The 15% → 74% jump came from correctly diagnosing the failure mode (overfitting on a long-tailed, low-sample-per-class dataset) rather than from simply training longer.

---

## 🔬 The Problem: Fine-Grained Classification with Sparse Data

| Property | Value |
|---|---|
| Classes | 683 plant species |
| Training images | 34,225 |
| Validation images | 2,679 |
| Avg. images per class (train) | ~50 |
| Smallest class | 1 sample |
| Largest class | 44 samples |

This is a **long-tailed, fine-grained** problem: many visually similar species, very few examples per class. A from-scratch CNN has no chance of learning general features from ~50 images per class — and indeed, the baseline hit 99.97% train accuracy while collapsing to 15% on validation.

---

## 🛠 Approach

**1. Baseline — Custom CNN**
Two conv blocks (Conv → BatchNorm → ReLU → MaxPool) + fully-connected head, trained with SGD (lr=0.002, momentum=0.9), Cross-Entropy loss, 10 epochs.
→ **Result: 15.04% val / 99.97% train.** Textbook overfitting.

**2. Diagnosis**
The train/val gap (~85 points) pointed at model capacity vastly exceeding the available signal per class — not at a training bug.

**3. Fixes applied**
- **Transfer learning** — ResNet-50 pretrained on ImageNet, replacing the classifier head for 683 classes
- **Data augmentation** — random horizontal flip, color jitter, random rotation (±20°)
- **Regularization** — Dropout(0.3) before the final linear layer
- **LR scheduling** — `ReduceLROnPlateau` (patience=3, factor=0.5) on validation loss

**4. Hyperparameter search**
Benchmarked optimizers and learning rates systematically:

| Optimizer | LR | Epochs | Best Val Acc |
|---|---|---|---|
| SGD (momentum=0.9) | 0.001 | 15 | 65.92% |
| SGD (momentum=0.9) | 0.01 | 15 | 64.13% |
| Adam | 0.001 | 20 | 56.77% |
| **Adam** | **0.0001** | **50** | **73.80%** ✅ |

Adam at a low learning rate combined with LR decay on plateau gave the best generalization — larger LRs destabilized the pretrained features.

---

## 🧰 Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `NumPy` · `Matplotlib` · `OpenCV` · `Jupyter`

---

## 📂 Repository

```
Herbarium-Classification/
├── Herbarium.ipynb    # Full pipeline: data → baseline → diagnosis → transfer learning → evaluation
└── README.md
```

The notebook is self-contained and runs end-to-end on Colab (GPU recommended).

---

## 💡 Key Takeaways

- **Overfitting is a data problem before it's a model problem.** With ~50 images across 683 fine-grained classes, no from-scratch architecture will generalize — pretrained features are not optional.
- **Learning rate matters more than optimizer choice.** Adam at 1e-3 underperformed SGD; Adam at 1e-4 beat everything.
- **Weighted F1 (0.68) is well below accuracy (0.74)**, reflecting the long tail — rare classes (some with a single sample) remain hard.

---

## 🚀 Future Work

- Class-balanced sampling or focal loss to address the long tail
- EfficientNet / ViT backbones for comparison
- Confusion matrix analysis on the most-confused species pairs
- Gradio demo for interactive inference

---

## 👤 Author

**Ali Alavi** — M.Sc. Computer Science, University of Messina
[LinkedIn](https://www.linkedin.com/in/ali-alavi-cs/) · [GitHub](https://github.com/salavii)
