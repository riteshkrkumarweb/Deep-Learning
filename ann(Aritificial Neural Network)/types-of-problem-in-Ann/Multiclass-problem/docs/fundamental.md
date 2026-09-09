# `reshape()` and `argmax()` 

## 1. `reshape()`

### Definition

`reshape()` changes the **shape (dimensions)** of an array **without changing its data**.

> **Memory Trick:**
> **reshape = Rearrange the same data**

### Example

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])

a.reshape(2,3)
```

Output:

```python
[[1 2 3]  The data is same only organizing is different , reshape(2,3) means 2 = row and 3 = columns 
 [4 5 6]]
```

The numbers are the same.

Only their arrangement changes.

### Real-Life Example

You have **6 books**.

Before:

```text
📚 📚 📚 📚 📚 📚
```

After arranging them on **2 shelves**:

```text
📚 📚 📚
📚 📚 📚
```

The books didn't change—only the arrangement changed.

---

## 2. `argmax()`

### Definition

`argmax()` returns the **index (position)** of the **largest value**.

> **Memory Trick:**
> **argmax = Argument (index) of the Maximum value**

### Example

```python
import numpy as np

a = np.array([10, 25, 15, 40, 20])

a.argmax()
```

Output:

```python
3
```

Why?

```text
Index : 0   1   2   3   4
Value :10  25  15  40  20
                  ↑
             Largest value
```

The largest value is **40**, and its **index is 3**.

So,

```python
a.argmax()
```

returns

```python
3
```

---

## In Deep Learning

Suppose the model predicts:

```python
prediction = [[0.01, 0.03, 0.90, 0.04, 0.02]]
```

Here, each value is the probability of a class.

Using:

```python
prediction.argmax(axis=1)
```

Output:

```python
array([2])
```

Because:

```text
Class :   0     1     2     3     4
Prob. : 0.01  0.03  0.90  0.04  0.02
                    ↑
             Highest Probability
```

The model predicts **Class 2**.

---

# Quick Summary

| Function    | Meaning                                               | Memory Trick                               |
| ----------- | ----------------------------------------------------- | ------------------------------------------ |
| `reshape()` | Changes the shape of data without changing the values | **Rearrange the same data**                |
| `argmax()`  | Returns the index of the largest value                | **Find the position of the maximum value** |

# `Flatten()`

## Definition

`Flatten()` converts **multi-dimensional data** into a **one-dimensional (1D) vector** without changing the data.

> **Memory Trick:**
> **Flatten = Make everything into one long line.**

---

## Example

Before:

```text
1 2 3
4 5 6
```

Shape:

```python
(2,3)
```

After `Flatten()`:

```text
1 2 3 4 5 6
```

Shape:

```python
(6,)
```

---

## MNIST Example

Before:

```python
(28,28)
```

After:

```python
Flatten()
```

Result:

```python
(784,)
```

because:

```text
28 × 28 = 784 pixels
```

---

## Why is `Flatten()` Used?

Dense (Fully Connected) layers accept **1D vectors**, not 2D images.

So `Flatten()` converts an image into a single vector before sending it to the Dense layer.

---

## Flow

```text
28×28 Image
      │
      ▼
  Flatten()
      │
      ▼
784 Values (1D)
      │
      ▼
 Dense Layer
```

---

## Quick Summary

| Function    | Purpose                                           |
| ----------- | ------------------------------------------------- |
| `Flatten()` | Converts multi-dimensional data into a 1D vector. |

### One-Line Memory Trick

> **Flatten = Convert everything into one long line (1D vector).**
