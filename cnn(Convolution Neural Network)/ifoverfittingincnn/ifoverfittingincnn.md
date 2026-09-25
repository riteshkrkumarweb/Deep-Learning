# Overfitting in CNN

## What is Overfitting?

Overfitting happens when a CNN learns the training data too well, including its specific details and noise, instead of learning general patterns.

The model performs very well on the training data but performs poorly on unseen validation or test data.

### Example

```text
Training Accuracy     → 99%
Validation Accuracy   → 75%

Training Loss         → Decreasing
Validation Loss       → Increasing
```

This is a common sign of overfitting.

---

## Why Does Overfitting Happen?

Common reasons:

- Small training dataset
- Very complex CNN
- Too many parameters
- Training for too many epochs
- Lack of regularization
- Training images are not diverse enough

---

## Solutions for Overfitting

### 1. Data Augmentation

Create realistic variations of training images.

Examples:

- Rotation
- Flipping
- Zooming
- Cropping
- Shifting
- Brightness changes

```text
Original Image
      ↓
 ┌────┼─────┬─────┐
 ↓    ↓     ↓     ↓
Flip Rotate Zoom  Crop
```

The model sees more variations and can learn more general features.

> Data augmentation does not create new independent images; it creates transformed versions of existing training images.

---

### 2. Dropout

Dropout randomly disables some neurons during training.

```python
layers.Dropout(0.5)
```

This prevents the network from depending too heavily on particular neurons.

---

### 3. L1 / L2 Regularization

Regularization adds a penalty to the model's weights.

Example of L2:

```python
layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=keras.regularizers.l2(0.001)
)
```

It encourages the model to use smaller weights and can help reduce overfitting.

---

### 4. Early Stopping

Stop training when validation performance stops improving.

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

Use it during training:

```python
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=50,
    callbacks=[early_stop]
)
```

---

### 5. More Training Data

A larger and more diverse dataset generally gives the CNN more examples from which to learn general patterns.

```text
Small dataset
     ↓
Easier to memorize
     ↓
Overfitting

Larger, diverse dataset
     ↓
More general patterns
     ↓
Better generalization
```

---

### 6. Reduce Model Complexity

A very large CNN may have more capacity than the dataset requires.

You can reduce:

- Number of layers
- Number of filters
- Number of Dense units
- Number of parameters

---

### 7. Transfer Learning

Use a pretrained CNN instead of training a large CNN completely from scratch.

```text
Pretrained CNN
      ↓
Already learned useful visual features
      ↓
Adapt to your dataset
```

This is especially useful when the dataset is relatively small.

---

### 8. Batch Normalization

Batch Normalization can stabilize training and may provide some regularizing effect.

```python
layers.Conv2D(32, 3, activation="relu"),
layers.BatchNormalization()
```

It can help, but it should not be considered a guaranteed solution by itself.

---

## How to Identify Overfitting

Look at the training and validation curves.

### Typical pattern

```text
Training Loss
      ↓
    decreases

Validation Loss
      ↓
    decreases initially
      ↓
    starts increasing
```

At the same time:

```text
Training Accuracy  ↑
Validation Accuracy ↓
```

This gap is a warning sign of overfitting.

---

## Quick Summary

| Technique | Main idea |
|---|---|
| Data Augmentation | Increase training-image variation |
| Dropout | Prevent neuron co-adaptation |
| L1/L2 | Penalize large weights |
| Early Stopping | Stop before excessive fitting |
| More Data | Provide more diverse examples |
| Reduce Complexity | Reduce model capacity |
| Transfer Learning | Reuse pretrained visual features |
| Batch Normalization | Stabilize training; may regularize |

## Key Idea

> **Overfitting means the CNN performs well on training data but does not generalize well to unseen data.**

The goal is not simply to make training accuracy higher. The goal is to make the model **generalize well to new images**.
