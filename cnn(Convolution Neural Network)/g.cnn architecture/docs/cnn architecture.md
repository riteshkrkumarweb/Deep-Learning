# 🧠 Convolutional Neural Network (CNN)

## 1. What is CNN?

**CNN (Convolutional Neural Network)** is a type of **Deep Neural Network (DNN)** mainly used for processing and understanding **images and other grid-like data**.

A CNN automatically learns important visual features such as:

* Edges
* Corners
* Textures
* Shapes
* Objects

Instead of manually defining image features, CNN learns them automatically using **convolution filters** during training.

### Basic CNN Architecture

```text
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Flatten
     ↓
Fully Connected (Dense)
     ↓
Output Layer
     ↓
Prediction
```

---

# 2. Full CNN Architecture

A typical CNN can be represented as:

```text
                    CNN ARCHITECTURE

┌───────────────────────────────┐
│         Input Image           │
│       28 × 28 × 1             │
│        (Grayscale)             │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│        Convolution            │
│      3 × 3 Filters             │
│     Learn image features       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│             ReLU              │
│      Remove negative values    │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│          Max Pooling           │
│            2 × 2               │
│     Reduce spatial size        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│        Convolution            │
│      3 × 3 Filters             │
│    Learn higher-level features │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│             ReLU              │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│          Max Pooling           │
│            2 × 2               │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│            Flatten             │
│   3D Feature Maps → 1D Vector  │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Fully Connected          │
│          Dense Layer           │
│      Combines the features     │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│        Output Layer            │
│      Softmax / Sigmoid         │
└───────────────┬───────────────┘
                ↓
          Final Prediction
```

---

# 3. Step-by-Step CNN Architecture

## Step 1 — Input Image

The **input layer** receives the image.

An image is represented as a tensor.

### Grayscale image

A grayscale image has one channel:

```text
Height × Width × Channels

28 × 28 × 1
```

### RGB image

An RGB image has three channels:

```text
Height × Width × Channels

224 × 224 × 3
```

The three channels are:

```text
R → Red
G → Green
B → Blue
```

Therefore, an RGB image is a **3D tensor**:

```text
Height × Width × 3
```

For example:

```text
32 × 32 × 3
```

means:

```text
32 rows
32 columns
3 color channels
```

---

# 4. Step 2 — Convolution

## Definition

**Convolution** is the operation where a small learnable matrix called a **filter (kernel)** slides over the input image and performs mathematical operations to detect features.

The filter learns features such as:

```text
Edges → Corners → Textures → Shapes → Objects
```

### Example

Input:

```text
5 × 5
```

Filter:

```text
3 × 3
```

The filter slides over the image:

```text
Input Image          Filter

┌─────────────┐      ┌───────┐
│ 1 2 3 4 5   │      │ 1 0 1 │
│ 6 7 8 9 0   │      │ 0 1 0 │
│ 1 2 3 4 5   │      │ 1 0 1 │
│ 6 7 8 9 0   │      └───────┘
│ 1 2 3 4 5   │
└─────────────┘
```

The filter performs element-wise multiplication and addition.

The result is stored in a **feature map**.

```text
Input Image
     +
   Filter
     ↓
Feature Map
```

---

# 5. Multiple Filters

A CNN does not normally use only one filter.

It uses **multiple filters**, and each filter can learn a different feature.

For example:

```text
Filter 1 → Vertical edges
Filter 2 → Horizontal edges
Filter 3 → Diagonal edges
Filter 4 → Texture
```

Therefore:

```text
Input Image
     ↓
Multiple Filters
     ↓
Multiple Feature Maps
```

If we use 32 filters:

```text
Output Channels = 32
```

---

# 6. Convolution with RGB Images

For an RGB image:

```text
Input:

Height × Width × 3
```

A convolution filter must also cover all 3 channels.

For example:

```text
Input:
32 × 32 × 3

Filter:
3 × 3 × 3
```

One filter produces **one feature map**.

If we use 64 filters:

```text
64 filters
↓
64 feature maps
```

So the output has:

```text
Height × Width × 64
```

---

# 7. Step 3 — ReLU Activation

After convolution, we usually apply an activation function.

The most commonly used activation function in CNN hidden layers is **ReLU**.

## ReLU Definition

ReLU stands for:

**Rectified Linear Unit**

Its formula is:

```text
ReLU(x) = max(0, x)
```

This means:

```text
Negative value → 0
Positive value → unchanged
```

### Example

```text
Input:

[-3, -1, 0, 2, 5]

ReLU:

[ 0,  0, 0, 2, 5]
```

Therefore:

```text
Convolution
     ↓
Feature Map
     ↓
ReLU
     ↓
Activated Feature Map
```

ReLU introduces **non-linearity**, allowing the network to learn complex patterns.

---

# 8. Step 4 — Pooling

## Definition

**Pooling** is a downsampling operation that reduces the spatial dimensions of feature maps while retaining important information.

The most commonly used pooling method is:

**Max Pooling**

### Example

Input feature map:

```text
4 × 4

┌───────┬───────┐
│ 1  3  │ 2  4  │
│ 5  6  │ 7  8  │
├───────┼───────┤
│ 9  2  │ 3  1  │
│ 4  5  │ 6  7  │
└───────┴───────┘
```

Using:

```text
2 × 2 Max Pooling
Stride = 2
```

We select the maximum value from each region:

```text
6   8
9   7
```

Output:

```text
2 × 2
```

### Pooling Pipeline

```text
Feature Map
     ↓
Max Pooling
     ↓
Smaller Feature Map
```

---

# 9. Why Pooling is Used

Pooling mainly helps to:

◇ Reduce spatial dimensions

◇ Reduce computation

◇ Reduce the number of values passed to later layers

◇ Make the representation somewhat less sensitive to small translations/shifts

### Important

Pooling can also cause information loss.

Therefore, pooling has both advantages and disadvantages.

---

# 10. Types of Pooling

## A. Max Pooling

Selects the **maximum value** from each region.

```text
[1  5]
[3  2]

Max = 5
```

It is the **most commonly used pooling method** in traditional CNN architectures.

---

## B. Average Pooling

Calculates the average value.

```text
[1  5]
[3  2]

Average = (1 + 5 + 3 + 2) / 4
        = 2.75
```

---

## C. Global Average Pooling

Instead of using a small window such as `2 × 2`, global average pooling calculates the average across the **entire spatial dimensions** of each feature map.

For example:

```text
7 × 7 × 64
```

can become:

```text
1 × 1 × 64
```

This can be used instead of a Flatten + large Dense layer in some CNN designs.

---

# 11. Step 5 — Second Convolution Block

After pooling, another convolution layer can be applied.

```text
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
```

The important idea is that CNN layers progressively learn more complex features.

### Early layers

```text
Edges
Lines
Simple textures
```

### Middle layers

```text
Corners
Shapes
Patterns
```

### Deeper layers

```text
Parts of objects
Complex shapes
Object-level patterns
```

---

# 12. Step 6 — Flatten

After the convolution and pooling layers, the feature maps are still usually represented as a multi-dimensional tensor.

Before passing them to a traditional Dense layer, we can flatten them.

## Definition

**Flattening converts a multi-dimensional feature map into a one-dimensional vector.**

Example:

```text
2 × 2 × 3
```

becomes:

```text
12 values
```

```text
Feature Maps

2 × 2 × 3
     ↓
   Flatten
     ↓
1 × 1 × 12
```

Conceptually:

```text
[
  [a b]
  [c d]
]

[
  [e f]
  [g h]
]

[
  [i j]
  [k l]
]

      ↓

[a b c d e f g h i j k l]
```

---

# 13. Step 7 — Fully Connected / Dense Layer

## Definition

A **Fully Connected (Dense) layer** connects each neuron in one layer to the neurons in the next layer.

It combines the features extracted by the convolutional layers to help make the final classification decision.

Example:

```text
Flatten
   ↓
[0.2, 0.8, 0.1, 0.6, ...]
   ↓
Dense Layer
   ↓
Learned combination of features
```

The convolution layers answer:

> "What features are present?"

The Dense layers help answer:

> "What does these features represent?"

---

# 14. Step 8 — Output Layer

The output layer produces the final prediction.

The activation function depends on the task.

## Multi-Class Classification

For multiple mutually exclusive classes, **Softmax** is commonly used.

Example:

```text
Cat       → 0.10
Dog       → 0.75
Horse     → 0.15
```

The probabilities sum to approximately:

```text
1.00
```

The highest probability is:

```text
Dog
```

Therefore:

```text
Prediction = Dog
```

---

# 15. Binary Classification

For two classes, a common setup is:

```text
Dense(1)
   ↓
Sigmoid
```

Example:

```text
Cat = 0.92
```

If the chosen threshold is 0.5:

```text
0.92 > 0.5

Prediction = Cat
```

---

# 16. Complete CNN Flow

The complete process can be represented as:

```text
                 IMAGE
                   ↓
          ┌─────────────────┐
          │  Input Layer    │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │  Convolution    │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │      ReLU       │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │   Max Pooling   │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │  Convolution    │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │      ReLU       │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │   Max Pooling   │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │     Flatten     │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │  Dense Layer    │
          └────────┬────────┘
                   ↓
          ┌─────────────────┐
          │  Output Layer   │
          │ Softmax/Sigmoid │
          └────────┬────────┘
                   ↓
              PREDICTION
```

---

# 17. Example: CNN for MNIST

Suppose we want to classify handwritten digits.

MNIST images are:

```text
28 × 28 × 1
```

A simple CNN could look like:

```text
Input
28 × 28 × 1
     ↓
Conv2D
32 filters, 3 × 3
     ↓
ReLU
     ↓
MaxPooling
2 × 2
     ↓
Conv2D
64 filters, 3 × 3
     ↓
ReLU
     ↓
MaxPooling
2 × 2
     ↓
Flatten
     ↓
Dense
128 neurons
     ↓
ReLU
     ↓
Dense
10 neurons
     ↓
Softmax
     ↓
Digit Prediction
0–9
```

The final output contains 10 class probabilities:

```text
0 → 0.01
1 → 0.02
2 → 0.01
3 → 0.03
4 → 0.01
5 → 0.04
6 → 0.02
7 → 0.80  ← highest
8 → 0.03
9 → 0.03
```

Therefore:

```text
Prediction = 7
```

---

# 18. What Happens During Training?

CNN does not manually decide what each filter should detect.

The filters contain **learnable weights**.

During training:

```text
Input Image
     ↓
CNN Forward Pass
     ↓
Prediction
     ↓
Compare with True Label
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Update Weights
     ↓
Better Filters
```

The network gradually learns useful filters.

For example, an early filter may learn to detect:

```text
Edges
```

while deeper layers can learn:

```text
Shapes → Object Parts → Objects
```

---

# 19. CNN Architecture in One Line

The basic CNN pipeline is:

```text
Input
→ Convolution
→ ReLU
→ Pooling
→ Convolution
→ ReLU
→ Pooling
→ Flatten
→ Dense
→ Output
→ Prediction
```

---

# 20. Important Terms

| Term            | Meaning                                                  |
| --------------- | -------------------------------------------------------- |
| Input           | Image given to the CNN                                   |
| Tensor          | Multi-dimensional numerical representation               |
| Channel         | Depth of image, e.g. 1 for grayscale and 3 for RGB       |
| Filter/Kernel   | Learnable matrix used to detect features                 |
| Convolution     | Operation that applies filters to the input              |
| Feature Map     | Output produced by a filter                              |
| ReLU            | Activation function that replaces negative values with 0 |
| Pooling         | Downsampling operation                                   |
| Max Pooling     | Selects maximum value from a region                      |
| Flatten         | Converts feature maps into a 1D vector                   |
| Dense           | Fully connected layer                                    |
| Softmax         | Converts logits into class probabilities                 |
| Sigmoid         | Common activation for binary classification              |
| Loss            | Measures prediction error                                |
| Backpropagation | Computes gradients for updating weights                  |
| Weights         | Learnable parameters of the network                      |

---

# 21. Final Summary

A CNN works in two major stages:

## Feature Extraction

```text
Input
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
```

The CNN learns useful visual features.

## Classification

```text
Feature Maps
     ↓
Flatten / Global Average Pooling
     ↓
Dense Layer(s)
     ↓
Output Layer
     ↓
Prediction
```

### In simple words:

```text
Convolution → Find features
ReLU        → Add non-linearity
Pooling     → Reduce spatial size
Flatten     → Convert to vector
Dense       → Combine learned features
Output      → Make prediction
```

> **CNN = Feature Extraction + Classification**

The most important idea is:

```text
Pixels
  ↓
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Object Parts
  ↓
Objects
  ↓
Prediction
```
