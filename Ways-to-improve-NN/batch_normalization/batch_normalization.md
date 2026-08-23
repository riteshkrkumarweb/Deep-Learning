# Batch Normalization

## 1. What is Batch Normalization?

**Batch Normalization (BN)** normalizes the activations of a neural network using the statistics of the current mini-batch.

**Main benefits:** more stable training, often faster convergence, and sometimes a regularization effect.

---

## 2. Distribution

**Distribution** means **how values are spread or arranged** in a dataset.

Example:

```text
10, 11, 12, 13, 14
```

Values are around `10–14`.

If they change to:

```text
100, 110, 120, 130, 140
```

the distribution has changed.

---

## 3. Covariate Shift

**Covariate Shift** means the distribution of the **input features** changes between training and testing, while the relationship between input and output remains approximately the same.

```text
Covariate Shift
→ Input distribution changes
```

---

## 4. Internal Covariate Shift

**Internal Covariate Shift** means the distribution of **internal activations** changes during training because the network's parameters keep changing.

```text
Covariate Shift
→ Input distribution changes

Internal Covariate Shift
→ Internal activation distribution changes
```

> The original BatchNorm paper motivated BN partly by reducing Internal Covariate Shift. Later research showed that this does not fully explain BN's effectiveness; improved optimization is an important part of its benefit.

---

## 5. How Batch Normalization Works

For each mini-batch:

```text
1. Calculate mean
        ↓
2. Calculate variance
        ↓
3. Normalize
        ↓
4. Scale using γ
        ↓
5. Shift using β
```


After normalization:

```text
Mean ≈ 0
Standard deviation ≈ 1
```

---

## 6. Gamma (γ) and Beta (β)

Normalization alone forces the values toward a standard scale, so BN uses two learnable parameters:

```text
γ → Scale
β → Shift
```

Final BN output:

```text
y = γ × z_norm + β
```

**Gamma:** controls the scale of normalized values.

**Beta:** controls the shift of normalized values.

Both are **trainable**.

---

## 7. Four Important BatchNorm Parameters/Statistics

| Parameter       | Trainable? | Purpose               |
| --------------- | ---------- | --------------------- |
| `γ` (Gamma)     | ✅ Yes      | Scale                 |
| `β` (Beta)      | ✅ Yes      | Shift                 |
| Moving Mean     | ❌ No       | Used during inference |
| Moving Variance | ❌ No       | Used during inference |

```text
γ → Trainable
β → Trainable

Moving Mean
Moving Variance
→ Non-trainable
```

Moving mean and moving variance are maintained during training and used during inference.

---

## 8. Training vs Testing

### During Training

```text
Current mini-batch
        ↓
Calculate mean + variance
        ↓
Normalize
        ↓
γ × value + β
        ↓
Update moving mean/variance
```

### During Testing/Inference

```text
New data
   ↓
Moving Mean + Moving Variance
   ↓
Normalize
   ↓
Prediction
```

---

## 9. Weight Initialization vs BatchNorm

**Weight Initialization** provides good starting values for weights.

```text
Xavier / He
     ↓
Good initial weights
     ↓
Better initial activation/gradient scale
```

**BatchNorm** controls activations during training.

```text
BatchNorm
    ↓
Normalize activations
    ↓
More stable optimization
```

> Weight initialization does not directly eliminate Internal Covariate Shift.

---

## 10. Advantages

* **Stable training** → keeps activation scales more controlled.

* **Often faster convergence** → optimization can become easier.

* **Can allow larger learning rates** → training can be less sensitive to the exact learning rate.

* **Some regularization effect** → batch statistics introduce noise that can sometimes improve generalization.

---

## 11. Why BN Can Sometimes Take More Time

BatchNorm adds extra calculations:

```text
Mean
Variance
Normalization
× γ
+ β
Moving statistics
```

Therefore, **time per epoch can sometimes be slightly higher** with BN.

```text
Without BN → 1.0 sec/epoch
With BN    → 1.1 sec/epoch
```

But BN may still reduce the **number of epochs needed for good performance**.

```text
Without BN → 100 epochs
With BN    → 60 epochs
```

So:

> **BN does not guarantee less time per epoch. Its main benefit is improved optimization and often faster convergence.**

---

## 12. BatchNorm vs Dropout

**BatchNorm:**

```text
Normalize activations
+
Improve optimization
+
Improve training stability
```

**Dropout:**

```text
Randomly deactivate neurons
→ Mainly used to reduce overfitting
```

They are different techniques.

---

## 13. Complete BatchNorm Flow

```text
Input
 ↓
Dense Layer
 ↓
Activation (z)
 ↓
Calculate Batch Mean + Variance
 ↓
Normalize
 ↓
× γ
 ↓
+ β
 ↓
BN Output
 ↓
Activation Function
 ↓
Next Layer
```

---

## 14. Quick Revision

```text
Batch Normalization
→ Normalize internal activations

Distribution
→ How values are spread/arranged

Covariate Shift
→ Input distribution changes

Internal Covariate Shift
→ Internal activation distribution changes

μ + σ²
→ Batch statistics

γ + β
→ Trainable scale and shift

Moving Mean + Moving Variance
→ Non-trainable running statistics

Training
→ Current batch statistics

Inference
→ Moving/running statistics

Main benefits
→ Stable training
→ Often faster convergence
→ Can allow larger learning rates
→ Some regularization effect
```
