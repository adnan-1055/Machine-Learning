---

> A full-semester practical implementation of core machine learning algorithms — from classical clustering and classification through to deep convolutional neural networks — applied to real datasets with rigorous performance analysis.

---

## Module Overview

| Detail | Info |
|---|---|
| **Module Code** | CP60057E |
| **Module Title** | Machine Learning |
| **Academic Year** | 2025–2026 |
| **Module Leader** | Professor Massoud Zolgharni |
| **Module Tutor** | Eman Alajrami |
| **Assessment Word Count** | ~2,500 (written answers) |

---

## Repository Structure

```
CP60057E-Machine-Learning/
│
├── Session-03-Clustering/
│   └── kmeans_elbow.py              # K-Means + Elbow Method
│
├── Session-04-Classification/
│   └── knn_classifier.py            # K-NN (K=3 and K=7) with ROC curves
│
├── Session-05-RandomForest/
│   └── random_forest_iris.py        # RF with 2, 3, 4 features per split
│
├── Session-06-Perceptron/
│   └── perceptron.py                # Linear Perceptron classifier
│
├── Session-07-SVM/
│   ├── svm_task1.py                 # SVM with C=[0.01, 0.1, 1.0]
│   └── svm_task2.py                 # Manual SVM boundary + support vectors
│
├── Session-08-NeuralNetworks/
│   └── mlp_iris.py                  # MLP architectures on Iris dataset
│
├── Session-09-ImageClassification/
│   └── knn_cifar10.py               # K-NN on CIFAR-10 (cross-validation)
│
├── Session-10-CNN/
│   └── cnn_cifar10.py               # CNN: epochs × kernel size experiments
│
├── Session-11-Regression/
│   └── linear_regression.py         # Linear regression + cost function J
│
├── report/
│   └── 33114153_Osman_MohammedAdnan.pdf
│
└── README.md
```

---

## Sessions & Key Results

### Session 03 — Clustering (K-Means)

Applied the **Elbow Method** to determine optimal K on a synthetic 2D dataset.

| K | WCSS | Decision |
|---|---|---|
| 1 | ~50,000 | Too few clusters |
| **3** | **~3,500** | **✅ Optimal — elbow point** |
| 4+ | Minimal drop | Diminishing returns |

**Key insight:** K-Means with K=3 produced three well-separated clusters with clearly defined centroids. Noted limitations: assumes spherical clusters, sensitive to centroid initialisation. DBSCAN flagged as a more robust alternative for non-spherical distributions.

---

### Session 04 — Classification (K-Nearest Neighbours)

K-NN tested on a binary classification dataset (300 test samples).

| Metric | K=3 | K=7 |
|---|---|---|
| Accuracy | 0.867 | **0.877** |
| Recall | 0.907 | **0.940** |
| F1 Score | 0.872 | **0.884** |
| AUC-ROC | 0.928 | **0.943** |
| Specificity | **0.827** | 0.813 |

**Winner: K=7** — higher AUC (0.943 vs 0.928) and F1, reflecting a better bias-variance trade-off. K=3 overfits training noise; K=7 smooths decision boundaries through local averaging.

---

### Session 05 — Random Forest (Iris Dataset)

Random Forest classifier tested with varying `max_features` on 150-sample Iris dataset (118 train / 32 test).

| max_features | Accuracy | Misclassifications |
|---|---|---|
| 2 | 93.75% | 2 Versicolour → Virginica |
| **3** | **96.88%** | **1 Versicolour → Virginica** |
| 4 | 93.75% | 2 Versicolour → Virginica |

**Winner: 3 features** — optimal balance between tree diversity and information gain. Using all 4 features removes ensemble randomness; using 2 limits split quality.

**Feature importance (consistent across all configurations):**
```
Petal Length  ████████████████████  ~54%
Petal Width   ████████████████████  ~44%
Sepal Length  ██                     ~1-8%
Sepal Width   █                      ~1-2%
```

---

### Session 06 — Perceptron

Single-layer Perceptron applied to a 1,000-point 2D binary classification dataset.

```
Final Weights:  W₀ = 25.000 | W₁ = 4.850 | W₂ = -15.210
Misclassified Points: 59 / 1000
Error Rate: 5.90%
```

Data is **not linearly separable** — class overlap near the decision boundary prevents zero-error convergence (Perceptron Convergence Theorem). SVM and MLP flagged as superior alternatives for overlapping distributions.

---

### Session 07 — Support Vector Machines

Linear SVM tested across regularisation values on the same 1,000-point dataset as Session 06.

| C | Accuracy | Error Rate | Support Vectors | Margin Width |
|---|---|---|---|---|
| 0.01 | 95.5% | 4.5% | 224 | 2.0595 |
| 0.10 | 95.7% | 4.3% | 131 | 1.3149 |
| **1.00** | **95.8%** | **4.2%** | 109 | 1.1158 |

**SVM vs Perceptron:** SVM achieves 95.8% vs Perceptron's 94.1% — margin maximisation (structural risk minimisation) reduces sensitivity to noise and improves generalisation.

**Task 2 — Manual SVM:**
```
Decision boundary: X₁ = 2  (vertical hyperplane)
Support Vectors:   (1,1), (1,5) [Class -1]  |  (3,3) [Class +1]
Parameters:        W = [1, 0],  b = -2,  Margin Width = 2
```

---

### Session 08 — Neural Networks (MLP on Iris)

Multi-Layer Perceptron architectures compared on Iris dataset.

| Architecture | Test Accuracy | Diagnosis |
|---|---|---|
| `(2,)` | 31.58% | Severe underfitting |
| `(4,4,2)` | **97.37%** | Best generalisation |
| `(20,10,5)` | 97.37% | No gain, higher cost |

**Key finding:** Network capacity must match data complexity. Adding layers/neurons beyond the task's complexity threshold yields no accuracy gain and increases overfitting risk. 100% test accuracy was not achieved (one sample always misclassified).

---

### Session 09 — Image Classification (K-NN on CIFAR-10)

K-NN with vectorised no-loop Euclidean distance, 5-fold cross-validation.

```
Optimal K: 11  (selected via cross-validation)
Test Accuracy: 32.4%  (3,240 / 10,000 images correct)
```

**Why so low?** Each image = 3,072 raw pixel features (32×32×3). In such high-dimensional spaces, Euclidean distance becomes meaningless (curse of dimensionality). CNNs with learned feature representations required for competitive performance.

---

### Session 10 — Convolutional Neural Networks (CIFAR-10)

**Task 1 — Manual convolution calculation:**
```
Source:  [[3,0,1],[2,6,2],[2,4,1]]
Filter:  [[-1,0,1],[-2,0,2],[-1,0,1]]  (Sobel edge detector)
Result:  -3
```

**Task 2 — CNN training experiments:**

| Kernel | Epochs | Test Accuracy |
|---|---|---|
| 3×3 | 10 | 0.6974 |
| 3×3 | 20 | **0.6992** ✅ |
| 3×3 | 50 | 0.6928 |
| 5×5 | 10 | 0.6750 |
| 5×5 | 20 | 0.6755 |
| 5×5 | 50 | 0.6642 |

**Findings:**
- Peak accuracy at **20 epochs** — beyond this, training accuracy diverges from validation (overfitting)
- **3×3 consistently outperforms 5×5** — smaller kernels need fewer parameters, preserve effective receptive fields when stacked, and suit CIFAR-10's 32×32 image size
- CNN (69.9%) massively outperforms K-NN (32.4%) by learning hierarchical spatial features rather than comparing raw pixels

---

### 🟤 Session 11 — Linear Regression

**Task 1 — Least Squares Error:**
```
Model: Ŷ = 6.49x + 30.18
LSE:   867.32
```

**Task 2 — Best model selection (3 candidates, Advertising dataset):**

| Model | Equation | LSE |
|---|---|---|
| Model 1 | Ŷ = 22.60x + 169.2 | 29,224 |
| **Model 2** | **Ŷ = 23.42x + 167.7** | **18,837 ✅** |
| Model 3 | Ŷ = 23.10x + 168.1 | 20,439 |

**Task 3 — Marketing Budget → Sales:**
```
Regression equation:  Ŷ = 10.55X + 2100.08

At £0 budget:         Expected sales = £2,100.08
At £7,500 budget:     Expected sales = £81,206.12
```

**Task 4 — Cost function J:**
```python
def compute_cost(X, y, m, b):
    total_cost = 0
    n = len(y)
    for i in range(n):
        y_hat = m * X[i] + b
        total_cost += (y_hat - y[i])**2
    return total_cost / (2 * n)

J = 2,172,668.16
```

---

## 🧠 Algorithm Comparison Summary

| Algorithm | Dataset | Best Accuracy | Key Strength |
|---|---|---|---|
| K-Means (K=3) | Synthetic 2D | — | Unsupervised grouping |
| K-NN (K=7) | Binary classification | 87.7% | Simple, no training phase |
| Random Forest (3 features) | Iris | 96.88% | Ensemble diversity |
| Perceptron | Binary 2D | 94.1% | Fast, linear problems |
| SVM (C=1.0) | Binary 2D | 95.8% | Margin maximisation |
| MLP (4,4,2) | Iris | 97.37% | Non-linear boundaries |
| K-NN | CIFAR-10 | 32.4% | Baseline only |
| CNN (3×3, 20ep) | CIFAR-10 | 69.92% | Spatial feature learning |

---

## 🛠️ Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

| Library | Usage |
|---|---|
| `scikit-learn` | K-NN, SVM, Random Forest, Perceptron, K-Means, MLP |
| `tensorflow` / `keras` | CNN architecture and training |
| `numpy` | Vectorised distance computation (K-NN no-loop) |
| `matplotlib` | All plots — ROC curves, confusion matrices, accuracy graphs |
| `pandas` | Dataset loading and preprocessing |

---

## Running the Scripts

```bash
# Clone the repository
git clone https://github.com/<your-username>/CP60057E-Machine-Learning.git
cd CP60057E-Machine-Learning

# Run any session script, e.g.:
python Session-03-Clustering/kmeans_elbow.py
python Session-10-CNN/cnn_cifar10.py
```

> **Note:** CIFAR-10 dataset is downloaded automatically via `keras.datasets.cifar10`. Iris dataset loaded via `sklearn.datasets.load_iris`.

---

## References

- Zolgharni, M. (2025) *Machine Learning: Google Colab Notebooks (Weeks 3–11)*. University of West London.
- Zolgharni, M. (2025) *Machine Learning: Lecture Slides (Weeks 1–11)*. University of West London.
- Géron, A. (2022) *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 3rd edn. O'Reilly Media.
- Kundu, R. (2022) *F1 Score in Machine Learning: Intro & Calculation*. V7labs.
- GeeksforGeeks (2025) *Curse of Dimensionality in Machine Learning*.
- GeeksforGeeks (2025) *Advantages and Disadvantages of Random Forest*.

---

<div align="center">

**Mohammed Adnan Osman · 33114153 · BSc Computer Science**
University of West London · CP60057E Machine Learning · 2025–2026

</div>
