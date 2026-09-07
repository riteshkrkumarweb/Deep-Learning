# CNN (Convolutional Neural Network)

## 1. What is CNN?

CNN stands for **Convolutional Neural Network**.

CNN is a type of neural network mainly used for **images and other grid-like data**.

The basic idea:

Image → Feature Extraction → Classification → Prediction

CNN automatically learns useful visual features such as:

Edges → Corners → Curves → Shapes → Object Parts → Complete Object

---

## 2. What Does CNN Stand For?

CNN:

C → Convolutional
N → Neural
N → Network

### Convolutional

It refers to the use of **convolution operations**.

A small matrix called a **filter/kernel** moves across the image and detects useful patterns.

### Neural

CNN is a type of **neural network** that learns weights and biases from data.

### Network

It consists of multiple connected layers that process the input and produce an output.

---

# 3. Why Do We Need CNN?

A normal ANN can process an image by flattening it:

Image → Flatten → ANN → Prediction

But an image contains **spatial information**.

For example:

Eye → Nose → Eye
       ↓
     Mouth

The position and relationship between nearby pixels are important.

CNN is designed to use this spatial information efficiently.

### Problems with using a normal ANN directly on images

- Very large number of parameters
- High computational cost
- Higher chance of overfitting
- Does not efficiently exploit spatial relationships

---

# 4. What is a Pixel?

A **pixel** is one small unit of a digital image.

An image is made up of many pixels.

Example:

■ ■ ■ ■
■ ■ ■ ■
■ ■ ■ ■
■ ■ ■ ■

Each square represents one pixel.

---

# 5. What Does 28 × 28 Mean?

When we say:

28 × 28 image

it means:

Height = 28 pixels
Width  = 28 pixels

It does NOT mean:

28 cm
28 inches
28 mm

It refers to the number of pixels.

Therefore:

28 × 28 = 784 pixels

So a 28 × 28 image contains **784 pixels**.

Example:

             Width = 28 pixels
        ←────────────────────→

        ■ ■ ■ ■ ■ ... ■
        ■ ■ ■ ■ ■ ... ■
        ■ ■ ■ ■ ■ ... ■
        ■ ■ ■ ■ ■ ... ■
        .
        .
        .
        ■ ■ ■ ■ ■ ... ■
        ↑
        Height = 28 pixels

---

# 6. Real Example of a 28 × 28 Image

A famous example is the **MNIST handwritten digit dataset**.

An MNIST image contains a handwritten digit such as:

7

The image size is:

Height = 28 pixels
Width  = 28 pixels

Therefore:

28 × 28 = 784 pixels

To the computer, the image is basically a **28 × 28 matrix of numbers**.

---

# 7. Grayscale Image

A grayscale image contains shades from black to white.

For a typical 8-bit grayscale image:

0   → Black
255 → White

Values between 0 and 255 represent different shades of gray.

Example:

0   → Black
50  → Dark Gray
128 → Gray
200 → Light Gray
255 → White

A small grayscale image can be represented as:

[
 [  0,  50, 120,  50,   0 ],
 [  0, 100, 200, 100,   0 ],
 [ 50, 180, 255, 180,  50 ],
 [  0, 100, 200, 100,   0 ],
 [  0,  50, 120,  50,   0 ]
]

Each number represents **one pixel**.

Therefore:

Image → Matrix → Pixel Values

---

# 8. What is a Channel?

A **channel** represents one component of image information.

A normal RGB color image has 3 channels:

R → Red
G → Green
B → Blue

Therefore, an RGB image has:

Height × Width × Channels

For example:

224 × 224 × 3

means:

Height   = 224 pixels
Width    = 224 pixels
Channels = 3

The 3 channels are:

Red Channel
Green Channel
Blue Channel

---

# 9. Grayscale vs RGB

## Grayscale

A grayscale pixel contains **one value**.

Example:

Pixel = 128

Image shape:

28 × 28

Total pixels:

28 × 28 = 784

---

## RGB

An RGB pixel contains **three values**:

Pixel = (R, G, B)

Examples:

(255, 0, 0) → Red
(0, 255, 0) → Green
(0, 0, 255) → Blue
(255, 255, 255) → White
(0, 0, 0) → Black

Image shape:

28 × 28 × 3

Total pixels:

28 × 28 = 784 pixels

Total numerical values:

28 × 28 × 3 = 2352 values

### IMPORTANT

2352 is NOT the number of pixels.

There are:

784 pixels

Each pixel contains:

3 values → R, G, B

Therefore:

784 pixels × 3 values = 2352 values

---

# 10. Real Example of an RGB Image

Consider a small 3 × 3 RGB image:

🟥 🟩 🟦
⬜ ⬛ 🟨
🟪 🟧 🟦

The computer can represent it as:

[
 [(255,0,0),     (0,255,0),     (0,0,255)],
 [(255,255,255), (0,0,0),       (255,255,0)],
 [(128,0,128),   (255,165,0),   (0,0,255)]
]

Each `(R,G,B)` represents **one pixel**.

For example:

(255, 0, 0)

R = 255
G = 0
B = 0

Therefore, this pixel is red.

---

# 11. RGB Channels as Separate Matrices

The RGB image can be separated into 3 matrices.

### Red Channel

[
 [255,   0,   0],
 [255,   0, 255],
 [128, 255,   0]
]

### Green Channel

[
 [  0, 255,   0],
 [255,   0, 255],
 [  0, 165,   0]
]

### Blue Channel

[
 [  0,   0, 255],
 [255,   0,   0],
 [128,   0, 255]
]

Together:

Red Channel
     +
Green Channel
     +
Blue Channel
     ↓
RGB Color Image

---

# 12. What Does Height × Width × Channels Mean?

For an RGB image:

28 × 28 × 3

means:

28 → Height
28 → Width
3  → Channels

So:

28 rows
×
28 columns
×
3 color values per pixel

There are:

28 × 28 = 784 pixels

Each pixel has:

3 values → R, G, B

Therefore:

784 × 3 = 2352 numerical values

---

# 13. Important Difference

### Grayscale

Image
 ↓
28 × 28
 ↓
784 pixels
 ↓
1 value per pixel

### RGB

Image
 ↓
28 × 28 × 3
 ↓
784 pixels
 ↓
3 values per pixel
 ↓
R + G + B

---

# 14. What is a Filter / Kernel?

A **filter**, also called a **kernel**, is a small learnable matrix used by CNN to detect patterns.

Example:

3 × 3 Filter

[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]

The filter moves across different parts of the image.

Image
┌─────────────────┐
│                 │
│   ┌─────┐       │
│   │3 × 3│       │
│   └─────┘       │
│                 │
└─────────────────┘

The filter moves:

→ → → → →

Filters can learn to detect:

Edges
Corners
Curves
Textures
Shapes

---

# 15. What is Convolution?

**Convolution** is the mathematical operation where a filter moves across the image and performs calculations using the image values and filter values.

Basic idea:

Image
  +
Filter / Kernel
  ↓
Mathematical Operation
  ↓
Feature Map

---

# 16. What is a Feature Map?

The output produced after applying a filter to an image is called a **feature map**.

Example:

Image
  ↓
Vertical-edge filter
  ↓
Feature Map

The feature map shows where the learned feature is present.

---

# 17. How Does CNN Learn Features?

CNN learns features in a hierarchy.

Input Image
     ↓
Simple Features
     ↓
Edges
     ↓
Corners / Curves
     ↓
Textures / Shapes
     ↓
Object Parts
     ↓
Complete Object

Example:

Pixels
 ↓
Edges
 ↓
Curves
 ↓
Eyes + Ears + Nose
 ↓
Face
 ↓
Cat

This is called **hierarchical feature learning**.

---

# 22. Key Points to Remember

- CNN = **Convolutional Neural Network**
- CNN is a type of neural network.
- CNN is mainly used for images and grid-like data.
- An image is made up of pixels.
- `28 × 28` means 28 pixels in height and 28 pixels in width.
- `28 × 28 = 784` pixels.
- 28 does not represent cm, inches, or another physical measurement.
- A grayscale pixel usually contains one intensity value.
- An RGB pixel contains three values: R, G, B.
- RGB images have three channels.
- `28 × 28 × 3` means 784 pixels with 3 values per pixel.
- `2352` is the number of numerical color values, not pixels.
- A filter/kernel is a small learnable matrix.
- Convolution applies the filter across the image.
- Convolution produces feature maps.
- CNN learns increasingly complex features through deeper layers.

---

# 23. One-Line Summary

CNN = Convolutional Neural Network

Image
 ↓
Pixels
 ↓
Channels
 ↓
Filters
 ↓
Convolution
 ↓
Feature Maps
 ↓
Feature Extraction
 ↓
Classification
 ↓
Prediction