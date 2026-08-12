# Dropout

**Definition:**
Dropout is a **regularization technique** that randomly and temporarily disables some neurons during each training batch to reduce **overfitting**.

## How Dropout Works

* During training, some neurons are randomly made inactive.
* A new random dropout pattern can be used for each batch.
* The same neuron can be active in one batch and inactive in another.
* Neurons are **not permanently removed**.
* Their weights and biases still exist in the model.

### Example

Suppose there are **4 neurons** and:

`p = 0.5`

One batch:

```text
N1 → Active
N2 → Disabled
N3 → Active
N4 → Disabled
```

Another batch can have a different pattern:

```text
N1 → Disabled
N2 → Active
N3 → Active
N4 → Disabled
```

## Important Point About Parameters

If:

* 4 input features
* 4 hidden neurons

Then:

```text
Weights = 4 × 4 = 16
Biases  = 4
Total   = 20 parameters
```

Applying `Dropout(0.5)` **does not reduce the model's parameters from 20 to 10**.

The parameters still exist; some neurons simply do not participate in that particular training step.

## Why Does Dropout Work?

Without dropout, the network may become too dependent on particular neurons or feature combinations.

With dropout:

```text
Batch 1 → some neurons OFF
Batch 2 → different neurons OFF
Batch 3 → different neurons OFF
       ↓
Network learns more robust patterns
       ↓
Less dependence on particular neurons
       ↓
Less overfitting
       ↓
Better generalization
```

## Dropout and Prediction

### During training

```text
Dropout → ON
Some neurons → temporarily OFF
Forward pass → Loss → Backpropagation → Weight update
```

### During prediction

```text
Dropout → OFF
All neurons → Active
Complete trained network → Prediction
```

The different dropout patterns are **not combined into separate networks**. The **same neural network** continuously updates its weights across batches.

## Dropout in Regression

The dropout mechanism is the same.

Example:

```text
Input → Hidden Layers → Dropout → Output
                              ↓
                         House price
                         ₹250,000
```

The output is usually a continuous value.

```python
Dense(1)
```

## Dropout in Classification

The dropout mechanism is also the same.

### Binary classification

```python
Dense(1, activation="sigmoid")
```

Example:

```text
0.87 → 87% probability of class 1
```

### Multi-class classification

```python
Dense(3, activation="softmax")
```

Example:

```text
[0.10, 0.75, 0.15]
```

## Regression vs Classification

|                   | Regression         | Classification     |
| ----------------- | ------------------ | ------------------ |
| Dropout           | Same technique     | Same technique     |
| Main purpose      | Reduce overfitting | Reduce overfitting |
| Output            | Continuous value   | Class/probability  |
| Output activation | Often none         | Sigmoid/Softmax    |

# Practical Tips

### 1. Dropout vs Overfitting

```text
Overfitting increases
        ↓
Try increasing dropout

Underfitting increases
        ↓
Try decreasing dropout
```

Don't treat this as a strict rule; adjust it based on **validation performance**.

### 2. Where to Apply Dropout

* Start with dropout in a **hidden layer**, often toward the later part of the network.
* If overfitting continues, try adding dropout to other hidden layers.
* You generally **do not need to put dropout on the output layer**.
* Test different placements and rates rather than automatically applying dropout everywhere.

### 3. Typical Starting Ranges

These are **rough starting points, not fixed rules**:

| Architecture |          Possible starting range |
| ------------ | -------------------------------: |
| ANN / MLP    |                       **10–50%** |
| CNN          | **40–50%** in some architectures |
| RNN          |         **20–30%** in some cases |

The best dropout rate depends on the dataset, architecture, layer, and amount of overfitting. Don't assume these percentages are optimal.

## Drawbacks of Dropout

◇ **Slower convergence:** Because the network sees a randomly changing set of active neurons during training, learning can take longer.

◇ **Noisy training:** The dropout mask changes from batch to batch, so the training loss can fluctuate more.

◇ **Too much dropout can cause underfitting:** If too many neurons are disabled, the network may not learn enough from the data.

◇ **Hyperparameter to tune:** The dropout rate and where dropout is applied can affect performance.

## Key Points

◇ `p` = probability of dropping a neuron.

◇ `1 - p` = probability of keeping a neuron.

◇ Dropout normally acts on the **layer where you add the Dropout layer**.

◇ Standard hidden-layer dropout drops **neurons**, not individual input features.

◇ Dropout does **not permanently delete neurons or weights**.

◇ Dropout can use a different random pattern for each training batch.

◇ During prediction, dropout is **OFF** and all neurons are available.

◇ Dropout can be used in both **regression and classification**.

**One-line summary:**

> **Dropout temporarily disables random neurons during training so the same neural network cannot rely too heavily on particular neurons, helping reduce overfitting and improve generalization.**
