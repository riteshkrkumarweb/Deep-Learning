# 🧠 Pretrained Models — CNN / Deep Learning Notes

## 1. What is a Pretrained Model?

A **pretrained model** is a neural network whose weights have already been learned on a large dataset.

Instead of starting with random weights:

```text
Random weights
     ↓
Train from scratch
     ↓
Learn edges → textures → shapes → objects
```

we can start with:

```text
Large dataset (e.g. ImageNet)
          ↓
     Train CNN
          ↓
  Pretrained model
          ↓
   Your own dataset
          ↓
 Transfer Learning
```

The main benefit is that the model has already learned useful visual representations.

---

## 2. Why Use Pretrained Models?

### ◇ 1. Less data required

Deep neural networks usually need a lot of data.

If your own dataset is relatively small, training a large CNN from scratch can be difficult.

A pretrained model gives you a useful starting point.

### ◇ 2. Less training time

You do not have to learn all visual features from zero.

### ◇ 3. Reuse learned features

CNNs learn hierarchical features:

```text
Early layers
   ↓
Edges / lines / corners

Middle layers
   ↓
Textures / patterns / shapes

Later layers
   ↓
Object parts / high-level features
```

These learned representations can often be useful for another related image task.

### ◇ 4. Useful for transfer learning

You can replace the original classifier and train the model for your own classes.

---

# 3. ImageNet

**ImageNet** is a large visual dataset/project containing images organized into many object categories.

It became an important dataset for computer-vision research and helped provide large-scale training and benchmarking data.

A simplified idea:

```text
ImageNet
   ↓
Large collection of labeled images
   ↓
Train CNN
   ↓
Learn visual representations
   ↓
Pretrained model
```

---

# 4. ILSVRC

**ILSVRC = ImageNet Large Scale Visual Recognition Challenge**

ILSVRC was a large-scale computer-vision benchmark/competition associated with ImageNet.

The commonly discussed classification task used **1,000 classes** and roughly **1.2 million training images**.

Important distinction:

```text
ImageNet
   ↓
Large dataset / project

ILSVRC
   ↓
Benchmark / challenge using a subset of ImageNet
```

Do not treat ImageNet and ILSVRC as exactly the same thing.

---

# 5. Classical Computer Vision vs Deep Learning

Before deep CNNs became dominant, image-recognition systems often used manually designed feature descriptors.

Examples:

```text
SIFT
HOG
```

Typical traditional pipeline:

```text
Image
  ↓
Hand-designed feature extraction
  ↓
SIFT / HOG / other features
  ↓
Machine Learning algorithm
  ↓
Prediction
```

Deep learning changed this:

```text
Image
  ↓
CNN
  ↓
Automatically learned features
  ↓
Prediction
```

### SIFT

**SIFT = Scale-Invariant Feature Transform**

It is a classical computer-vision method for finding and describing distinctive local image features.

### HOG

**HOG = Histogram of Oriented Gradients**

It represents information about local gradient/edge directions and was widely used in traditional object-detection systems.

---

# 6. Important CNN Architectures

Several architectures became important in the development of modern computer vision.

```text
AlexNet
   ↓
ZFNet
   ↓
VGG
   ↓
GoogLeNet / Inception
   ↓
ResNet
```

### AlexNet

AlexNet was a landmark deep CNN architecture that achieved a major improvement in the 2012 ILSVRC image-classification competition.

Its success helped demonstrate the effectiveness of deep CNNs for large-scale visual recognition.

### VGG

VGG is a family of CNN architectures known for using many small convolution filters, commonly 3×3 filters, in a deep network.

Popular versions include:

```text
VGG16
VGG19
```

### GoogLeNet / Inception

Inception architectures use modules containing different operations in parallel.

Conceptually:

```text
             Input
               ↓
       ┌───────┼────────┐
       ↓       ↓        ↓
      1×1     3×3      5×5
      Conv    Conv     Conv
       ↓       ↓        ↓
       └───────┼────────┘
               ↓
             Output
```

### ResNet

**ResNet = Residual Network**

ResNet introduced residual/skip connections.

```text
Input ───────────────────┐
  ↓                      │
Conv → Conv              │
  ↓                      │
  └──────── + ←──────────┘
            ↓
          Output
```

Skip connections help very deep networks learn effectively.

---

# 7. What is Transfer Learning?

**Transfer learning** means using knowledge learned from one task/dataset to help solve another related task.

For example:

```text
ImageNet
   ↓
Pretrained ResNet
   ↓
Your dog-breed dataset
   ↓
Dog-breed classifier
```

The pretrained model provides a starting point instead of random initialization.

---

# 8. Two Main Ways to Use a Pretrained Model

## A. Feature Extraction

Use the pretrained CNN as a fixed feature extractor.

```text
Pretrained CNN
      ↓
Freeze layers
      ↓
Extract features
      ↓
New classifier
      ↓
Your prediction
```

The pretrained layers are not updated during training.

---

## B. Fine-Tuning

First freeze the pretrained model and train your new classifier.

Then optionally unfreeze some of the pretrained layers and continue training with a **small learning rate**.

```text
Pretrained model
      ↓
Freeze base model
      ↓
Train new classifier
      ↓
Unfreeze some upper layers
      ↓
Small learning rate
      ↓
Fine-tune
```

Fine-tuning allows the learned representations to adapt to your dataset.

---

# 9. Keras Applications

Keras provides many architectures with optional pretrained weights.

Examples include:

```text
ResNet
VGG
Inception
MobileNet
DenseNet
Xception
NASNet
```

Keras Applications can be used for:

```text
Prediction
Feature extraction
Transfer learning
Fine-tuning
```

Official documentation:

- Keras Applications: https://keras.io/api/applications/
- Transfer Learning & Fine-Tuning: https://keras.io/guides/transfer_learning/

---

# 10. Load a Pretrained ResNet50

```python
import keras

# Load ResNet50 with weights already trained on ImageNet.
model = keras.applications.ResNet50(
    weights="imagenet"
)

model.summary()
```

### Explanation

```python
weights="imagenet"
```

means:

> Load weights that were pretrained on ImageNet.

If you use:

```python
weights=None
```

the model starts with randomly initialized weights.

---

# 11. Image Classification with ResNet50

```python
import numpy as np
import keras

from keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions
)

# Load pretrained ResNet50.
model = ResNet50(weights="imagenet")

# Path to your image.
img_path = "elephant.jpg"

# Load and resize image to the size expected by ResNet50.
img = keras.utils.load_img(
    img_path,
    target_size=(224, 224)
)

# Convert PIL image to NumPy array.
x = keras.utils.img_to_array(img)

# Add batch dimension.
# Shape changes from:
# (224, 224, 3)
# to:
# (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Apply ResNet50-specific preprocessing.
x = preprocess_input(x)

# Make prediction.
preds = model.predict(x)

# Convert ImageNet class IDs into readable class names.
print(decode_predictions(preds, top=3)[0])
```

### Pipeline

```text
Image
 ↓
Resize
 ↓
Convert to array
 ↓
Add batch dimension
 ↓
Preprocess
 ↓
ResNet50
 ↓
Prediction
 ↓
Decode ImageNet classes
```

---

# 12. What does `include_top` mean?

Consider:

```python
model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False
)
```

The original ImageNet classification head is removed.

Conceptually:

```text
             ResNet
               │
      Convolutional base
               │
               X
        ImageNet classifier
          (removed)
```

This is useful when you want to attach your own classifier.

---

# 13. Feature Extraction Example

```python
import keras
from keras import layers

# Load pretrained ResNet50 without the original ImageNet classifier.
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers.
base_model.trainable = False

# Create a new input.
inputs = keras.Input(shape=(224, 224, 3))

# Run input through pretrained CNN.
x = base_model(inputs, training=False)

# Convert feature maps into one vector per image.
x = layers.GlobalAveragePooling2D()(x)

# Add your own classifier.
outputs = layers.Dense(
    10,
    activation="softmax"
)(x)

# Create complete model.
model = keras.Model(inputs, outputs)

model.summary()
```

Here:

```python
base_model.trainable = False
```

means the pretrained CNN weights are frozen.

Only the newly added classifier is trained.

---

# 14. Binary Classification Example

Suppose your dataset contains:

```text
cats
dogs
```

Then you can create:

```python
import keras
from keras import layers

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers.
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

# Keep the pretrained model in inference mode.
x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

# One output for binary classification.
outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

---

# 15. Multi-Class Classification Example

Suppose you have:

```text
10 classes
```

Use:

```python
outputs = layers.Dense(
    10,
    activation="softmax"
)(x)
```

The number of output neurons must correspond to your number of classes when using this setup.

For example:

```text
10 classes → Dense(10)
20 classes → Dense(20)
```

---

# 16. Fine-Tuning

After training the new classifier, you can optionally fine-tune part of the pretrained network.

```python
# Unfreeze the base model.
base_model.trainable = True

# Freeze most of the early layers.
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile after changing trainable settings.
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Fine-tune the model.
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=5
)
```

### Why use a small learning rate?

The pretrained model already contains useful learned weights.

A large learning rate could change those weights too aggressively.

So fine-tuning generally uses a smaller learning rate.

---

# 17. Important: Preprocessing

Different pretrained architectures can require different preprocessing.

For example, ResNet50 uses:

```python
keras.applications.resnet50.preprocess_input
```

VGG16 uses:

```python
keras.applications.vgg16.preprocess_input
```

InceptionV3 uses:

```python
keras.applications.inception_v3.preprocess_input
```

Always check the documentation for the specific architecture you are using.

---

# 18. `weights="imagenet"` vs `weights=None`

### ImageNet pretrained

```python
model = keras.applications.ResNet50(
    weights="imagenet"
)
```

```text
Start
 ↓
ImageNet-learned weights
 ↓
Ready for prediction / transfer learning
```

### Random initialization

```python
model = keras.applications.ResNet50(
    weights=None
)
```

```text
Start
 ↓
Random weights
 ↓
Train from scratch
```

---

# 19. Feature Extraction vs Fine-Tuning

| Feature Extraction | Fine-Tuning |
|---|---|
| Freeze pretrained base | Unfreeze some/all layers |
| Faster training | More training required |
| Fewer trainable parameters | More trainable parameters |
| Good starting approach | Used to adapt representations |
| New classifier learns | Pretrained layers also adapt |

A common workflow is:

```text
1. Load pretrained model
        ↓
2. Freeze base
        ↓
3. Add classifier
        ↓
4. Train classifier
        ↓
5. Optionally unfreeze some layers
        ↓
6. Fine-tune with small learning rate
```

---

# 20. Important Keras Parameters

## `weights`

```python
weights="imagenet"
```

Loads ImageNet pretrained weights.

```python
weights=None
```

Uses random initialization.

---

## `include_top`

```python
include_top=True
```

Keep the original classification head.

```python
include_top=False
```

Remove the original classification head.

For transfer learning, `include_top=False` is commonly useful because you want to add your own classifier.

---

## `input_shape`

Example:

```python
input_shape=(224, 224, 3)
```

means:

```text
Height = 224
Width  = 224
Channels = 3
```

---

## `pooling`

When `include_top=False`, Keras can optionally apply global pooling:

```python
pooling="avg"
```

or:

```python
pooling="max"
```

---

# 21. Common Mistakes

### ◇ Forgetting preprocessing

Different architectures can expect different preprocessing.

### ◇ Using the wrong number of output classes

If your dataset has 5 classes:

```python
Dense(5, activation="softmax")
```

### ◇ Fine-tuning immediately with a large learning rate

This can damage useful pretrained representations.

### ◇ Forgetting to recompile after changing `trainable`

After changing which layers are trainable, compile the model again before training.

### ◇ Assuming every pretrained model uses the same input size/preprocessing

They do not necessarily do so.

---

# 22. The Complete Mental Model

Remember this:

```text
                 ImageNet
                    ↓
             Large-scale training
                    ↓
              CNN learns features
                    ↓
             Pretrained model
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
 Feature Extraction       Fine-Tuning
          ↓                   ↓
 Freeze base             Unfreeze some layers
          ↓                   ↓
 Train classifier         Small LR
          ↓                   ↓
 Your task                Your task
```

---

# 23. Short Summary ⭐

```text
Pretrained Model
=
A model whose weights were already learned on a large dataset.

ImageNet
=
A large visual dataset/project commonly used for pretraining.

ILSVRC
=
A major large-scale image-recognition benchmark associated with ImageNet.

Transfer Learning
=
Reuse learned features from a pretrained model for a new task.

Feature Extraction
=
Freeze the pretrained base and train a new classifier.

Fine-Tuning
=
Unfreeze some pretrained layers and train them further with a small learning rate.

include_top=False
=
Remove the original ImageNet classifier so you can add your own.

weights="imagenet"
=
Load ImageNet-pretrained weights.
```

---

# 24. One Code Template to Remember

```python
import keras
from keras import layers

# 1. Load pretrained model.
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Freeze pretrained layers.
base_model.trainable = False

# 3. Create your input.
inputs = keras.Input(shape=(224, 224, 3))

# 4. Extract features using the pretrained CNN.
x = base_model(inputs, training=False)

# 5. Convert feature maps into a vector.
x = layers.GlobalAveragePooling2D()(x)

# 6. Add your own classifier.
outputs = layers.Dense(
    10,
    activation="softmax"
)(x)

# 7. Build the final model.
model = keras.Model(inputs, outputs)

# 8. Compile.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 9. Train on your own dataset.
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
```

### Code flow

```text
Image
 ↓
Preprocessing
 ↓
Pretrained ResNet50
 ↓
Feature extraction
 ↓
GlobalAveragePooling
 ↓
Your Dense layer
 ↓
Your classes
```

---

## 📚 Official Documentation

◇ Keras Applications: https://keras.io/api/applications/

◇ Keras Transfer Learning & Fine-Tuning: https://keras.io/guides/transfer_learning/

◇ Keras ResNet: https://keras.io/api/applications/resnet/resnet_models/

◇ Keras VGG: https://keras.io/api/applications/vgg/vgg_models/

◇ Keras InceptionV3: https://keras.io/api/applications/inceptionv3/inception_v3_model/

◇ Keras MobileNet: https://keras.io/api/applications/mobilenet/mobilenet_models/

---

## 🎯 Final Takeaway

The most important idea is:

```text
Train from scratch:

Random weights
     ↓
Learn everything
     ↓
Your model


Transfer Learning:

ImageNet pretrained weights
          ↓
Already learned useful features
          ↓
Adapt to your dataset
          ↓
Your model
```

**Pretrained models save you from having to learn useful visual representations completely from zero.**
