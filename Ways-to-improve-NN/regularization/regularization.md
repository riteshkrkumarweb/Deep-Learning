# Regularization in Deep Learning

Regularization → Reduces overfitting by penalizing large weights.

## L1 Regularization

Penalizes the absolute value of weights → Some weights become **0** → Creates a **sparse model**.

## L2 Regularization

Penalizes the squared value of weights → Weights become **smaller** → Makes the model more stable and reduces overfitting.

## Sparse Model

A model where **many weights are 0**, so those features have little or no contribution.

## When to Use

* **L1** → When you want **feature selection / sparse model**.
* **L2** → When you want to **keep features but reduce overfitting**.

## Memory

**L1 → Zero weights → Sparse**

**L2 → Small weights → Stable**
