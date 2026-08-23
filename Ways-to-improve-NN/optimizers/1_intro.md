# Introduction to Optimizers

## 1. Introduction to Optimizers

An **optimizer** is an algorithm used to update the weights and biases of a neural network so that the **loss function decreases**.

The basic idea is:

```text
Prediction
    ↓
Calculate Loss
    ↓
Calculate Gradient
    ↓
Optimizer
    ↓
Update Weights
    ↓
Repeat
```

The optimizer controls **how the model moves toward a lower-loss region**.

---

## 2. Role of Optimizers

The main roles of an optimizer are:

◇ **Update weights and biases** to reduce the loss.

◇ **Choose the direction of the update** using gradients.

◇ **Control the size of the update** using the learning rate.

◇ **Improve training speed and stability**.

Basic Gradient Descent:

```text
w_new = w_old - η × gradient
```

Where:

```text
η = learning rate
```

The optimizer uses the gradient to determine how the weights should change.

---

## 3. Types of Optimizers

### Gradient Descent Variants

◇ **Batch Gradient Descent**
Uses the entire training dataset to calculate one update.

◇ **Stochastic Gradient Descent (SGD)**
Uses one training sample for each update.

◇ **Mini-Batch Gradient Descent**
Uses a small batch of samples for each update.


---

## 4. Challenges of Optimization

Training a neural network can be difficult because of:

◇ **Learning Rate**

Too small → training becomes slow.
Too large → training may overshoot or become unstable.

◇ **Learning-Rate Scheduling**

The learning rate may need to change during training to make learning more efficient.

◇ **Complex Loss Landscape**

Neural networks can have complicated loss surfaces with valleys, hills, flat regions, and saddle points.

◇ **Local Minima**

The optimizer can reach a low point that is not necessarily the lowest point of the entire loss landscape.

◇ **Saddle Points**

The gradient can be close to zero even though the point is not a minimum, which can slow optimization.

◇ **Different Directions**

Different weights can have very different gradients, making optimization difficult.

---

## 5. What Next?


### Advanced Optimizers

◇ **Momentum**
Uses previous updates to build momentum and reduce unnecessary oscillations.

◇ **NAG (Nesterov Accelerated Gradient)**
Uses momentum and looks ahead before calculating the gradient.

◇ **AdaGrad**
Adapts the learning rate separately for different parameters.

◇ **RMSprop**
Uses a moving average of squared gradients to adapt the learning rate.

◇ **Adam**
Combines ideas from **Momentum and RMSprop** and is widely used in deep learning.

**Main goal:** Understand how each optimizer improves the basic Gradient Descent update.