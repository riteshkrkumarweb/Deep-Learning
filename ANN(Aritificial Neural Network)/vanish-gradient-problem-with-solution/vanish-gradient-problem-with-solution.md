# Vanishing Gradient Problem

## Definition
The **Vanishing Gradient Problem** occurs when the **gradient (partial derivative of the loss function)** becomes **very small** during backpropagation. As a result, the weights in the earlier layers receive tiny updates and learn very slowly or stop learning.

---

# Solutions to the Vanishing Gradient Problem

## 1. Reduce Model Complexity
* Use fewer hidden layers.
* Fewer layers → fewer multiplications of the gradient → larger gradients.

## 2. Use ReLU Activation Function
* ReLU derivative is **1** for positive inputs.
* Helps keep the gradient from becoming too small.

## 3. Proper Weight Initialization
* Initialize weights using **Xavier (Glorot)** or **He Initialization**.
* Keeps the gradient in a healthy range.

## 4. Batch Normalization
* Normalizes the outputs of each layer.
* Improves the flow of the gradient through the network.

## 5. Residual Networks (ResNet)
* Uses skip (shortcut) connections.
* Allows the gradient to flow directly to earlier layers, reducing the vanishing gradient problem.

---
# The Other topic in further like this but opposite is Exploding Gradient
The exploding gradient problem occurs when the gradients become **extremely large** during backpropagation. As a result, the weights change by huge amounts, making the model unstable and preventing it from learning properly.


**Gradient = Partial Derivative of the Loss Function**

* Very small gradient → **Vanishing Gradient**
* Very large gradient → **Exploding Gradient**