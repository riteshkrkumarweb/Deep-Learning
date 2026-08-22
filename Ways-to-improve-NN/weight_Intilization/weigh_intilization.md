# Weight Initialization in Neural Networks

## 1. Why Weight Initialization Matters

Weight initialization means assigning initial values to the weights before training.

Good initialization helps:

* Break symmetry between neurons
* Keep activations at a reasonable scale
* Keep gradients at a reasonable scale
* Make training faster and more stable

Poor initialization can cause:

* Vanishing gradients
* Exploding gradients
* Slow convergence
* Symmetry problems

---

## 2. Zero Initialization

Initializing all hidden-layer weights to zero is **not recommended**.

Why?

* All neurons start with the same weights.
* They produce the same outputs.
* They receive the same gradients.
* They continue learning the same features.

This is called the **symmetry problem**.

> Biases can generally be initialized to zero.

---

## 3. Constant Initialization

Example:

```python
W = 0.5
```

Using the same non-zero value for every weight also causes the **symmetry problem**.

The problem is not specifically the value `0`.

The problem is:

> All neurons have identical initial weights.

---

## 4. Random Initialization

Random initialization gives different neurons different starting weights.

This:

* Breaks symmetry
* Allows neurons to learn different features

But the random values should have an **appropriate scale**.

---

## 5. Very Small Random Weights

If weights are extremely small:

* Signals can become too small through deep layers.
* Gradients can become very small.
* Learning can become slow.
* It can contribute to the vanishing-gradient problem.

---

## 6. Very Large Random Weights

If weights are extremely large:

* Activations can become very large.
* Sigmoid/Tanh can enter saturation.
* Gradients can become very small.
* Training can become unstable.
* Large weights can also contribute to exploding gradients.

---

## 7. Vanishing Gradient

The gradient becomes extremely small as it is propagated backward through the network.

Result:

* Earlier layers receive tiny updates.
* Weights barely change.
* Learning becomes very slow.

---

## 8. Exploding Gradient

The gradient becomes extremely large.

Result:

* Very large weight updates
* Unstable training
* Loss may become extremely large or fail to converge

---

## 9. Saturation

Sigmoid and Tanh can become almost flat at their extreme values.

When the activation is saturated:

* Gradient becomes very small.
* Learning becomes slow.

---

# 10. Xavier / Glorot Initialization

Xavier initialization is commonly used with:

* **Tanh**
* **Sigmoid**

Its purpose is to keep activations and gradients at a reasonable scale.

### Xavier has two options:

* **Xavier Normal**
* **Xavier Uniform**

### Xavier Normal

Weights are sampled from a **normal (bell-shaped) distribution**.

### Xavier Uniform

Weights are sampled from a **uniform distribution**, meaning values are evenly spread within a suitable range.

You only need to choose **one**.

---

# 11. He Initialization

He initialization is commonly used with:

* **ReLU**
* **Leaky ReLU**
* Other ReLU-family activations

Its purpose is to maintain a suitable scale of activations and gradients when using ReLU-type activations.

### He also has two options:

* **He Normal**
* **He Uniform**

### He Normal

Weights are sampled from a normal distribution.

### He Uniform

Weights are sampled from a uniform distribution.

Again, choose **one**.

---

# 12. Normal vs Uniform Distribution

### Normal Distribution

* Bell-shaped
* Values near the center are more common.
* No strict minimum or maximum.

### Uniform Distribution

* Values are spread evenly within a fixed range.
* No value within that range is preferred over another.

> **Normal vs Uniform = how the random weights are distributed.**

---

# 13. Xavier vs He

| Activation | Recommended Initialization |
| ---------- | -------------------------- |
| ReLU       | He                         |
| Leaky ReLU | He                         |
| Tanh       | Xavier                     |
| Sigmoid    | Xavier                     |

Each has two variants:

| Initialization | Options          |
| -------------- | ---------------- |
| Xavier         | Normal / Uniform |
| He             | Normal / Uniform |

---

## 14. Final Cheat Sheet

```text
Weight Initialization
│
├── Zero weights ❌
│   └── Symmetry problem
│
├── Same constant weights ❌
│   └── Symmetry problem
│
├── Very small random weights ⚠️
│   └── Can cause weak signals / vanishing gradients
│
├── Very large random weights ❌
│   └── Saturation / unstable gradients
│
└── Proper random initialization ✅
    │
    ├── Xavier / Glorot
    │   ├── Normal
    │   └── Uniform
    │
    └── He
        ├── Normal
        └── Uniform
```

### 🧠 Remember

> **ReLU → He**
> **Tanh/Sigmoid → Xavier**
> **Normal or Uniform → choose one**
> **Hidden-layer weights should not all start with the same value.**

# Cheat Seat 
```text
Weight Initialization
        │
        ├── Zero ❌
        │     └── Symmetry
        │
        ├── Same constant ❌
        │     └── Symmetry
        │
        ├── Very small random ⚠️
        │     └── Weak signal / vanishing gradients
        │
        ├── Very large random ❌
        │     └── Saturation / instability / gradients problems
        │
        └── Proper random initialization ✅
              │
              ├── Xavier / Glorot(uniform/normal)
              │       └── Tanh / Sigmoid
              │
              └── He(uniform/normal)
                      └── ReLU / Leaky ReLU