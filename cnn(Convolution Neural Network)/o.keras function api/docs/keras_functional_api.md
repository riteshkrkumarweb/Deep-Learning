# Keras Functional API

## 1. Problem with the Sequential Model

The **Sequential API** is limited to a simple linear stack of layers, where each layer follows one path. It becomes unsuitable for complex architectures such as **multiple inputs, multiple outputs, branching, merging, and skip connections**.

```text
Input → Layer → Layer → Layer → Output
```

When the model architecture is not a simple linear chain, we use the **Keras Functional API**.

---

## 2. Why We Use the Keras Functional API

The **Keras Functional API** is a flexible way to build neural networks by explicitly defining how inputs, layers, and outputs are connected.

It supports:

- Multiple inputs
- Multiple outputs
- Branching
- Merging
- Skip connections
- Complex neural-network architectures

Basic structure:

```python
inputs = keras.Input(...)

x = Layer1()(inputs)
x = Layer2()(x)

outputs = Layer3()(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)
```

---

# 3. Single Input → Multiple Outputs

### Architecture

```text
                    ┌──→ Output 1
                    │
Input → Shared Layers
                    │
                    └──→ Output 2
```

### Example

One image is used to predict **age** and **gender**.

```python
from tensorflow import keras
from tensorflow.keras import layers

# Input
inputs = keras.Input(
    shape=(224, 224, 3),
    name="image"
)

# Shared feature extraction
x = layers.Conv2D(
    32,
    3,
    activation="relu"
)(inputs)

x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

# Output 1: Age
age_output = layers.Dense(
    1,
    name="age"
)(x)

# Output 2: Gender
gender_output = layers.Dense(
    1,
    activation="sigmoid",
    name="gender"
)(x)

# Create model
model = keras.Model(
    inputs=inputs,
    outputs=[age_output, gender_output]
)

model.summary()
```

### Structure

```text
                    ┌──→ Age
                    │
Image → CNN → Dense ┤
                    │
                    └──→ Gender
```

---

# 4. Multiple Inputs → Single Output

### Architecture

```text
Input 1 ──→ Network 1 ──┐
                        │
                        ├──→ Combine → Output
                        │
Input 2 ──→ Network 2 ──┘
```

### Example

Use **image + product metadata** to predict product price.

```python
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# Input 1: Image
# -------------------------

image_input = keras.Input(
    shape=(224, 224, 3),
    name="image"
)

image_features = layers.Conv2D(
    32,
    3,
    activation="relu"
)(image_input)

image_features = layers.MaxPooling2D()(
    image_features
)

image_features = layers.Flatten()(
    image_features
)

image_features = layers.Dense(
    64,
    activation="relu"
)(image_features)


# -------------------------
# Input 2: Metadata
# -------------------------

metadata_input = keras.Input(
    shape=(10,),
    name="metadata"
)

metadata_features = layers.Dense(
    32,
    activation="relu"
)(metadata_input)

metadata_features = layers.Dense(
    16,
    activation="relu"
)(metadata_features)


# -------------------------
# Combine both inputs
# -------------------------

combined = layers.Concatenate()([
    image_features,
    metadata_features
])


# -------------------------
# Final output
# -------------------------

x = layers.Dense(
    64,
    activation="relu"
)(combined)

price_output = layers.Dense(
    1,
    name="price"
)(x)


# -------------------------
# Create model
# -------------------------

model = keras.Model(
    inputs=[image_input, metadata_input],
    outputs=price_output
)

model.summary()
```

### Structure

```text
Image ───────→ CNN ────────────┐
                               │
                               ├──→ Concatenate → Dense → Price
                               │
Metadata ─────→ Dense ─────────┘
```

---

# 5. Multiple Inputs → Multiple Outputs

### Architecture

```text
Input 1 ──→ Network 1 ──┐
                         │
Input 2 ──→ Network 2 ──┼──→ Shared Features
                         │         │
                         │         ├──→ Output 1
                         │         │
                         │         └──→ Output 2
```

### Example

Use **image + metadata** to predict **age and gender**.

```python
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# Input 1: Image
# -------------------------

image_input = keras.Input(
    shape=(224, 224, 3),
    name="image"
)

image_features = layers.Conv2D(
    32,
    3,
    activation="relu"
)(image_input)

image_features = layers.MaxPooling2D()(
    image_features
)

image_features = layers.Flatten()(
    image_features
)

image_features = layers.Dense(
    64,
    activation="relu"
)(image_features)


# -------------------------
# Input 2: Metadata
# -------------------------

metadata_input = keras.Input(
    shape=(10,),
    name="metadata"
)

metadata_features = layers.Dense(
    32,
    activation="relu"
)(metadata_input)

metadata_features = layers.Dense(
    16,
    activation="relu"
)(metadata_features)


# -------------------------
# Combine inputs
# -------------------------

combined = layers.Concatenate()([
    image_features,
    metadata_features
])

shared = layers.Dense(
    64,
    activation="relu"
)(combined)


# -------------------------
# Output 1: Age
# -------------------------

age_output = layers.Dense(
    1,
    name="age"
)(shared)


# -------------------------
# Output 2: Gender
# -------------------------

gender_output = layers.Dense(
    1,
    activation="sigmoid",
    name="gender"
)(shared)


# -------------------------
# Create model
# -------------------------

model = keras.Model(
    inputs=[image_input, metadata_input],
    outputs=[age_output, gender_output]
)

model.summary()
```

### Structure

```text
                         ┌──→ Age
                         │
Image ──→ CNN ───────────┤
                         │
                         ├──→ Shared Features
                         │
Metadata ──→ Dense ──────┤
                         │
                         └──→ Gender
```

---

# 6. Quick Summary

| Architecture | Sequential API | Functional API |
|---|---|---|
| Single Input → Single Output | ✅ | ✅ |
| Single Input → Multiple Outputs | ❌ / Not suitable | ✅ |
| Multiple Inputs → Single Output | ❌ / Not suitable | ✅ |
| Multiple Inputs → Multiple Outputs | ❌ / Not suitable | ✅ |
| Branching | ❌ | ✅ |
| Merging | ❌ | ✅ |
| Skip Connections | ❌ | ✅ |

## Remember

```text
Sequential API
    ↓
Simple linear architecture

Functional API
    ↓
Flexible graph-based architecture
```

The key idea of Functional API is:

> **You explicitly define the connections between tensors, instead of forcing the model into one sequential chain.**
