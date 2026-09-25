# Transfer Learning

## What is Transfer Learning?

**Transfer Learning** is a technique where we use a model that is already trained on a large dataset and reuse its learned knowledge for a new, related task.

Instead of training a deep learning model from scratch, we start with a **pre-trained model**.

### Example

A CNN trained on ImageNet has already learned features such as:

```text
Edges → Textures → Shapes → Parts → Objects
```

We can reuse these features for a new task such as:

```text
Cat vs Dog Classification
```

---

## Why Use Transfer Learning?

- Requires less training data
- Training is faster
- Requires less computational power
- Often gives better results than training from scratch
- Useful when our dataset is small

---

# Types of Transfer Learning

## 1. Feature Extraction

The pre-trained model is used as a **fixed feature extractor**.

The pre-trained layers are frozen, so their weights do not change.

```text
Pre-trained Model
       ↓
Frozen Layers 🔒
       ↓
Features
       ↓
New Classifier 🔓
       ↓
Prediction
```

Example:

```python
base_model.trainable = False
```

Use this when the new task is similar to the original task and the dataset is small.

---

## 2. Fine-Tuning

In fine-tuning, some layers of the pre-trained model are **unfrozen** and trained on the new dataset.

```text
Early Layers     → Frozen 🔒
Later Layers     → Trainable 🔓
Classifier       → Trainable 🔓
```

Example:

```python
base_model.trainable = True
```

Usually, a **small learning rate** is used so that the pre-trained knowledge is not destroyed.

```python
optimizer = keras.optimizers.Adam(learning_rate=1e-5)
```

---

## 3. Partial Fine-Tuning

Only some of the later layers are unfrozen.

```python
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False
```

This means:

```text
Early layers → Frozen 🔒
Last 20 layers → Trainable 🔓
New classifier → Trainable 🔓
```

---

# Feature Extraction vs Fine-Tuning

| Feature Extraction | Fine-Tuning |
|---|---|
| Base model is frozen | Some layers are unfrozen |
| Faster training | Slower training |
| Fewer parameters trained | More parameters trained |
| Good for small datasets | Useful when more adaptation is needed |
| Uses existing features | Adapts features to the new task |

---

# Common Transfer Learning Workflow

```text
Pre-trained Model
       ↓
Remove Original Classifier
       ↓
Add New Classifier
       ↓
Freeze Base Model
       ↓
Train New Classifier
       ↓
Unfreeze Some Layers
       ↓
Fine-Tune
```

---

# Important Terms

### Pre-trained Model
A model that has already been trained on a large dataset.

Examples:

- VGG16
- VGG19
- ResNet50
- MobileNet
- EfficientNet
- Xception

### Frozen Layer

A layer whose weights are not updated during training.

```python
layer.trainable = False
```

### Trainable Layer

A layer whose weights can be updated during training.

```python
layer.trainable = True
```

---

# Key Idea

> **Transfer Learning = Reuse knowledge from a pre-trained model for a new task.**

In CNNs, early layers usually learn general features such as **edges and textures**, while later layers learn more task-specific features.

So we can reuse the early knowledge and fine-tune later layers for our new problem.
