# 🌿 Herbarium Classification — Fine-Grained Plant Species Recognition

Deep learning pipeline for classifying digitized herbarium specimens across **683 plant species**, built with PyTorch. The project walks through a full diagnosis-and-fix cycle: a custom CNN baseline that overfits catastrophically, then a transfer-learning solution with augmentation and regularization that recovers **~5× the validation accuracy**.

Dataset: [Kaggle Herbarium 2019 (FGVC6)](https://www.kaggle.com/c/herbarium-2019-fgvc6) —
but see [Reproducing this](#-reproducing-this) before assuming the notebook runs
from that download alone. It does not.

---

## 📊 Results

| Model | Train Acc | Best Val Acc | Notes |
|---|---|---|---|
| Custom CNN (2-layer, SGD) | 99.97% | **15.04%** | Severe overfitting — memorized the training set |
| ResNet-50 + augmentation + dropout | 99.99% | **73.80%** | Best model (Adam, lr=1e-4, LR scheduling) |

> **How to read these numbers.** They are the *best* epoch's validation accuracy,
> not the last epoch's — the final epoch of the ResNet-50 run scored 73.57%. There
> is no held-out test set: the run has only a train and a validation split, so the
> same data used to pick the best epoch is the data being reported on. That makes
> 73.80% mildly optimistic as an estimate of performance on unseen specimens, in
> the usual model-selection way. It is fine for comparing the configurations below
> against each other, which is what it is used for here.

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
| Avg. images per class (train) | 50.1 |
| Avg. images per class (validation) | 3.9 |
| Smallest class (validation) | 1 sample |
| Largest class (validation) | 44 samples |

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

## 🔁 Reproducing this

**The notebook will not run end-to-end from a clean checkout, and the honest
reason is that one step is missing from this repository.**

`Herbarium.ipynb` does not read the Kaggle competition data directly. It mounts
Google Drive and unzips two pre-built archives:

```python
drive.mount('/content/gdrive')
!unzip gdrive/My\ Drive/herbarium/small-train.zip
!unzip gdrive/My\ Drive/herbarium/small-validation.zip
```

Those archives are a **subset** of Herbarium 2019, prepared before the notebook
starts. **The code that built them was never committed here**, and it is not
recoverable from this repository's history — every commit has only ever
contained `Herbarium.ipynb` and `README.md`. Rather than guess at the procedure
after the fact, it is recorded as unknown.

What can be stated, because the notebook measures it directly:

| Property of the subset | Value |
|---|---|
| Classes | 683 (directories named `0`–`682`) |
| Training images | 34,225 |
| Validation images | 2,679 |
| Layout | `ImageFolder`-compatible: one directory per class ID |

What is **not** recorded: which 683 of the competition's species were kept, how
images were allocated between train and validation, whether any resizing or
filtering was applied, and what random seed (if any) drove the selection.

So the results below are **reproducible only against that exact subset**, not
from the Kaggle download alone. Anyone rebuilding a subset by a different rule
should expect different numbers, and the comparison between configurations —
which is the actual point of the project — will still hold within their own
subset.

**Setup, for the record:**

```bash
pip install -r requirements.txt
```

Then place the two extracted directories at `small-train/` and
`small-validation/` and point `train_dt` / `valid_dt` at them. A GPU is strongly
recommended — the final configuration is 50 epochs of ResNet-50.

Note also that the notebook seeds NumPy (`np.random.seed(1234)`) but never seeds
PyTorch, so weight initialisation, shuffling, and augmentation vary between runs.
Expect the numbers to move by a fraction of a point even on identical data.

**One cell has no stored output.** The class-average cell carried a bug: its
counts list sat at module scope and was never reset between calls, so the second
call appended to the first and reported the average over train *and* validation
combined (27.02) instead of validation's own (3.92). The bug is fixed, but the
stored output was produced by the old code, and filling in a corrected number by
hand would present output the cell never actually printed — so it was cleared.
Re-run that cell against the data to repopulate it. The correct values are in the
table above, and both are plain arithmetic: 34,225 / 683 and 2,679 / 683.

---

## 🧰 Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `NumPy` · `Matplotlib` · `OpenCV` · `Jupyter`

---

## 📂 Repository

```
Herbarium-Classification/
├── Herbarium.ipynb    # Full pipeline: data → baseline → diagnosis → transfer learning → evaluation
├── requirements.txt   # Hand-written from what the notebook actually calls
├── LICENSE
└── README.md
```

The notebook runs end-to-end on Colab with a GPU, **given the prepared data
subset** — see [Reproducing this](#-reproducing-this).

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

**Mohaddeseh Jahanabadi** — M.Sc. Computer Science, University of Messina
[LinkedIn](https://www.linkedin.com/in/mohaddeseh-jahanabadi/) · [GitHub](https://github.com/jahanabadi-n)

**Ali Alavi** — M.Sc. Computer Science, University of Messina
[LinkedIn](https://www.linkedin.com/in/ali-alavi-cs/) · [GitHub](https://github.com/salavii)

