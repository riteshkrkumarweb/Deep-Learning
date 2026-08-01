# 📘 Deep Learning Loss Functions – Revision Notes

## 🎯 What is a Loss Function?

A **Loss Function** measures how much the model's prediction differs from the actual value.

**Goal:** Minimize the loss by updating the model's weights and biases.

---

# 1. Mean Squared Error (MSE)

### Used For
* Regression

### Formula

Single Sample

\[
(y-\hat{y})^2
\]

Dataset

\[
MSE=\frac1n\sum(y-\hat{y})^2
\]

### Intuition

Squares the error, so **large errors get much higher punishment**.

Example

```
Error = 2 → Loss = 4
Error = 5 → Loss = 25
```

### Advantages

* Easy to compute
* Differentiable
* Smooth optimization

### Disadvantages

* Sensitive to outliers
* Loss is in squared units

---

# 2. Mean Absolute Error (MAE)

### Used For

* Regression

### Formula

Single Sample

\[
|y-\hat{y}|
\]

Dataset

\[
MAE=\frac1n\sum|y-\hat{y}|
\]

### Intuition

Uses the **absolute value**, so positive and negative errors do not cancel each other.

Example

```
+5 → 5
-5 → 5
```

### Advantages

* Robust to outliers
* Same unit as target
* Easy to understand

### Disadvantages

* Not differentiable at 0
* Training may be slower

---

# 3. Huber Loss

### Used For

* Regression with outliers

### Formula

\[
L=
\begin{cases}
\frac12(y-\hat y)^2,& |y-\hat y|\le\delta\\
\delta|y-\hat y|-\frac12\delta^2,& |y-\hat y|>\delta
\end{cases}
\]

### Intuition

```
Small Error
      ↓
     MSE

Large Error
      ↓
     MAE
```

### Advantages

* Less sensitive to outliers
* Differentiable
* Combines MSE + MAE

### Disadvantages

* Need to choose δ (delta)
* Slightly more complex

---

# 4. Binary Cross Entropy (BCE)

### Also Called

* Log Loss

### Used For

* Binary Classification (2 Classes)

Examples

* Yes / No
* Spam / Not Spam

### Activation Function

* Sigmoid

### Formula

\[
-\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right]
\]

### Intuition

Good prediction → Small loss

Wrong confident prediction → Large loss

### Advantages

* Best for binary classification
* Smooth gradients
* Differentiable

### Disadvantages

* Only for binary classification
* Requires Sigmoid output

---

# 5. Categorical Cross Entropy (CCE)

### Used For

* Multi-Class Classification (3+ Classes)

### Activation Function

* Softmax

### Label Format

* One-Hot Encoding

Example

```
Yes

↓

[1 0 0]
```

### Formula

\[
-\sum y\log(\hat y)
\]

### Advantages

* Best for multi-class classification
* Works perfectly with Softmax
* Differentiable

### Disadvantages

* Requires One-Hot Encoding
* Uses more memory

---

# 6. Sparse Categorical Cross Entropy (SCCE)

### Used For

* Multi-Class Classification

### Activation Function

* Softmax

### Label Format

* Integer Labels

Example

```
Yes → 0
No → 1
Maybe → 2
```

### Formula

\[
-\log(\hat y_{\text{correct class}})
\]

### Advantages

* No One-Hot Encoding required
* Faster
* Less memory
* Same loss as CCE

### Disadvantages

* Labels must be integers
* Cannot use One-Hot labels

---

# 📊 Summary Table

| Loss Function | Problem | Activation | Label Format |
|--------------|---------|------------|--------------|
| MSE | Regression | Linear | Continuous Values |
| MAE | Regression | Linear | Continuous Values |
| Huber | Regression | Linear | Continuous Values |
| Binary Cross Entropy | Binary Classification | Sigmoid | 0 / 1 |
| Categorical Cross Entropy | Multi-Class Classification | Softmax | One-Hot Encoding |
| Sparse Categorical Cross Entropy | Multi-Class Classification | Softmax | Integer Labels |

---

# 🚀 Which Loss Function Should I Use?

```
Problem
│
├── Regression
│   ├── Clean Data → MSE
│   ├── Many Outliers → MAE
│   └── Few Outliers → Huber
│
└── Classification
    ├── Binary (2 Classes)
    │      ├── Sigmoid
    │      └── Binary Cross Entropy
    │
    └── Multi-Class (3+ Classes)
           ├── One-Hot Labels
           │      └── Categorical Cross Entropy
           │
           └── Integer Labels
                  └── Sparse Categorical Cross Entropy
```

---

# 🧠 Memory Trick

```
MSE   → Squares Error

MAE   → Absolute Error

Huber → MSE + MAE

BCE   → Binary Classification

CCE   → Multi-Class + One-Hot Labels

SCCE  → Multi-Class + Integer Labels
```

---

# ⚡ 30-Second Revision

```
Regression
──────────
MSE   → Clean Data
MAE   → Many Outliers
Huber → Few Outliers

Binary Classification
─────────────────────
Sigmoid
↓
Binary Cross Entropy

Multi-Class Classification
──────────────────────────
One-Hot Labels
↓
Categorical Cross Entropy

Integer Labels
↓
Sparse Categorical Cross Entropy
```