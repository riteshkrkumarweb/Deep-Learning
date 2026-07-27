# Forward Propagation (MLP)

## Definition
Forward Propagation is the process of passing a **training example (input)** through every layer of a neural network to compute the final prediction (**ŷ**).

> During forward propagation, **no learning happens**. The network only uses the current weights and biases to make a prediction.

---

# Purpose

- Convert input features into a prediction.
- Pass information from the input layer to the output layer.
- Compute the activation of every neuron.
- Produce the final output (**ŷ**).

---

# Forward Propagation Flow

```text
Training Example (x)
        │
        ▼
Input Layer
        │
        ▼
Weighted Sum (Wx + b)
        │
        ▼
Activation Function σ()
        │
        ▼
Output (Activation)
        │
        ▼
Next Layer
        │
        ▼
Repeat...
        │
        ▼
Final Prediction (ŷ)
```

---

# Components of Forward Propagation

## 1. Training Example (Input)

A single sample from the dataset.

Example

```text
x =
[x₁
 x₂
 x₃
 x₄]
```

Shape

```text
4 × 1
```

---

## 2. Weight (W)

Weights determine the importance of each input feature.

- Every connection has one weight.
- Learned during training.
- Stored as a weight matrix.

Example

```text
Input (4) → Hidden (3)

Weights

4 × 3 = 12
```

---

## 3. Bias (b)

Each neuron has one bias.

Purpose

- Shifts the weighted sum.
- Makes the model more flexible.
- Helps fit data better.

Example

```text
Hidden Layer (3 neurons)

Biases = 3
```

---

## 4. Weighted Sum

Every neuron first computes

```text
z = Wx + b
```

where

```text
z = weighted sum
```

---

## 5. Activation Function

Applies non-linearity.

```text
a = σ(z)
```

Common activation functions

- Sigmoid
- ReLU
- Tanh
- Softmax (Output layer for multiclass)

---

## 6. Output (Activation)

The activation becomes the input for the next layer.

```text
a = Output of the neuron
```

---

# General Equation

For every layer

### Step 1

```text
z[l] = W[l]ᵀ a[l−1] + b[l]
```

### Step 2

```text
a[l] = σ(z[l])
```

where

```text
a⁰ = x
```

---

# Example Architecture

```text
Input (4)

↓

Hidden Layer 1 (3)

↓

Hidden Layer 2 (2)

↓

Output Layer (1)
```

---

# Layer 1 (Input → Hidden 1)

Equation

```text
a¹ = σ(W¹ᵀx + b¹)
```

Shapes

```text
Input        : 4 × 1

Weights      : 4 × 3

Weightsᵀ     : 3 × 4

Bias         : 3 × 1

Output (a¹)  : 3 × 1
```

---

# Layer 2 (Hidden 1 → Hidden 2)

Equation

```text
a² = σ(W²ᵀa¹ + b²)
```

Shapes

```text
Input (a¹)   : 3 × 1

Weights      : 3 × 2

Weightsᵀ     : 2 × 3

Bias         : 2 × 1

Output (a²)  : 2 × 1
```

---

# Layer 3 (Hidden 2 → Output)

Equation

```text
ŷ = a³ = σ(W³ᵀa² + b³)
```

Shapes

```text
Input (a²)   : 2 × 1

Weights      : 2 × 1

Weightsᵀ     : 1 × 2

Bias         : 1 × 1

Output (ŷ)   : 1 × 1
```

---

# Matrix Dimensions

| Connection | Weight Matrix | Bias | Output |
|------------|--------------|------|--------|
| Input → Hidden 1 | 4 × 3 | 3 × 1 | 3 × 1 |
| Hidden 1 → Hidden 2 | 3 × 2 | 2 × 1 | 2 × 1 |
| Hidden 2 → Output | 2 × 1 | 1 × 1 | 1 × 1 |

---

# Counting Weights

Formula

```text
Weights

=

(Number of neurons in previous layer)

×

(Number of neurons in current layer)
```

Example

```text
Input → Hidden 1

4 × 3 = 12
```

```text
Hidden 1 → Hidden 2

3 × 2 = 6
```

```text
Hidden 2 → Output

2 × 1 = 2
```

Total

```text
12 + 6 + 2 = 20 Weights
```

---

# Counting Biases

Formula

```text
Biases

=

Number of neurons in current layer
```

Example

```text
Hidden Layer 1

3 Biases
```

```text
Hidden Layer 2

2 Biases
```

```text
Output Layer

1 Bias
```

Total

```text
3 + 2 + 1 = 6 Biases
```

---

# Total Trainable Parameters

Formula

```text
Total Parameters

=

Weights + Biases
```

Layer-wise

```text
Input → Hidden 1

12 + 3 = 15
```

```text
Hidden 1 → Hidden 2

6 + 2 = 8
```

```text
Hidden 2 → Output

2 + 1 = 3
```

Overall

```text
15 + 8 + 3 = 26 Parameters
```

---

# Complete Forward Pass

```text
Training Example (x)

↓

Weighted Sum (Wx + b)

↓

Activation Function

↓

Output (a)

↓

Repeat for every hidden layer

↓

Prediction (ŷ)
```

---

# Intuition

Think of each neuron as a small decision-maker.

Each neuron:

1. Receives inputs from the previous layer.
2. Multiplies each input by its weight.
3. Adds one bias.
4. Applies an activation function.
5. Sends its output to the next layer.

The last neuron produces the final prediction (**ŷ**).

---

# Symbols

| Symbol | Meaning |
|---------|---------|
| **x** | Training Example / Input |
| **W** | Weight Matrix |
| **b** | Bias Vector |
| **z** | Weighted Sum |
| **σ** | Activation Function |
| **a** | Activation (Neuron Output) |
| **ŷ** | Final Prediction |
| **l** | Layer Number |

---

# Key Points

- Forward propagation moves information from **input → output**.
- Every connection has one **weight**.
- Every neuron has one **bias**.
- Every neuron computes **z = Wx + b**.
- Activation is computed as **a = σ(z)**.
- The output of one layer becomes the input to the next layer.
- Matrix dimensions must match for multiplication.
- Total parameters = **Weights + Biases**.
- Forward propagation only computes predictions.
- Learning (weight updates) happens during **Backpropagation**.
```