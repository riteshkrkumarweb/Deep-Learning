# Multi-Layer Perceptron (MLP) — Summary Notes

## 1. What is an MLP?

A **Multi-Layer Perceptron (MLP)** is a type of **Feedforward Neural Network (FNN)**(A Feedforward Neural Network (FNN) is a neural network in which information flows only from the input layer, through the hidden layer(s), to the output layer, without any feedback loops) that consists of an **input layer**, **one or more hidden layers**, and an **output layer**. It learns both **linear and non-linear relationships** using activation functions.

---

## 2. Why MLP?

The Perceptron can only solve **linear problems**, while an MLP can solve **complex non-linear problems** such as XOR, image classification, speech recognition, and many real-world tasks.

---

## 3. Architecture of an MLP

```text
Input Layer
      │
      ▼
Hidden Layer(s)
      │
      ▼
Output Layer
```

---

## 4. Layers in an MLP

### Input Layer

* Receives input features.
* Performs no computation.

### Hidden Layer

* Learns patterns and feature representations.
* Each neuron computes:

  * Weighted Sum
  * Activation Function

### Output Layer

* Produces the final prediction.
* Activation depends on the problem type.

---

## 5. Working of an MLP

```
Input Features
      ↓
Weighted Sum (z = w·x + b)
      ↓
Activation Function
      ↓
Hidden Layer(s)
      ↓
Output Layer
      ↓
Prediction
```

---

## 6. Weighted Sum

Each neuron computes:

[
z = w \cdot x + b
]

where:

* **w** = Weights
* **x** = Input Features
* **b** = Bias
* **z** = Linear Combination

---

## 7. Activation Function

Activation functions introduce **non-linearity**, allowing the MLP to learn complex patterns.

Common activations:

* ReLU
* Sigmoid
* Tanh
* Softmax
* Linear

---

## 8. Output Layer Activation

| Problem                    | Activation |
| -------------------------- | ---------- |
| Binary Classification      | Sigmoid    |
| Multi-Class Classification | Softmax    |
| Regression                 | Linear     |

---

## 9. Forward Propagation

Forward propagation is the process of passing data from the **input layer** to the **output layer** to generate predictions.

```
Input
   ↓
Hidden Layer
   ↓
Output
```

---

## 10. Learning Process

1. Initialize weights and bias.
2. Perform forward propagation.
3. Calculate loss.
4. Compute gradients using backpropagation.
5. Update weights using gradient descent.
6. Repeat until the loss is minimized.

---

## 11. Advantages of MLP

* Learns complex non-linear relationships.
* Solves XOR and other difficult problems.
* Works for both classification and regression.
* Foundation of modern deep learning.

---

## 12. Limitations of MLP

* Requires large datasets.
* Computationally expensive.
* Can overfit.
* Requires hyperparameter tuning.

---

# Quick Revision

| Concept                    | Summary                                       |
| -------------------------- | --------------------------------------------- |
| MLP                        | Feedforward neural network with hidden layers |
| Input Layer                | Receives input features                       |
| Hidden Layer               | Learns patterns using neurons                 |
| Output Layer               | Produces prediction                           |
| Weighted Sum               | (z = w \cdot x + b)                           |
| Activation Function        | Adds non-linearity                            |
| Forward Propagation        | Input → Hidden → Output                       |
| Backpropagation            | Computes gradients                            |
| Gradient Descent           | Updates weights                               |
| Binary Classification      | Sigmoid                                       |
| Multi-Class Classification | Softmax                                       |
| Regression                 | Linear                                        |
