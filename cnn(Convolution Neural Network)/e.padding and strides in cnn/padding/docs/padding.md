# 🧩 Padding in CNN

## What is Padding?

**Padding** is the process of adding extra pixels around the border of an image before applying a convolution filter.

The most common type is **zero padding**, where the added pixels have a value of `0`.

### Example

Original image:

```text
1  1  1  1  1
1  1  1  1  1
1  1  1  1  1
1  1  1  1  1
1  1  1  1  1
```

After adding 1-pixel zero padding:

```text
0  0  0  0  0  0  0
0  1  1  1  1  1  0
0  1  1  1  1  1  0
0  1  1  1  1  1  0
0  1  1  1  1  1  0
0  1  1  1  1  1  0
0  0  0  0  0  0  0
```

The original image was:

```text
5 × 5
```

After padding:

```text
7 × 7
```

The image itself has not changed. We have simply added a border around it.

---

## Why is Padding Used?

◇ **Preserves edge information**

Without padding, pixels near the edges of an image are used less during convolution.

Padding allows the filter to operate around the image boundaries.

◇ **Prevents unnecessary shrinking**

Without padding, applying a convolution filter generally makes the feature map smaller.

Padding can keep the feature map at the same spatial size.

◇ **Allows edge pixels to participate more**

The added border gives the filter space to process areas close to the image boundaries.

---

# Types of Padding

There are two commonly used padding options in Keras:

```text
valid
same
```

## 1. `padding='valid'`

`valid` means:

> **No padding is added.**

Example:

```text
Input:
5 × 5

Filter:
3 × 3

Padding:
None

Output:
3 × 3
```

Keras:

```python
Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding='valid'
)
```

Or simply:

```python
Conv2D(32, 3, padding='valid')
```

So:

```text
padding='valid'
        ↓
No padding
```

---

# 2. `padding='same'`

`same` means:

> **Keras automatically adds the required padding so the output has the same height and width as the input in the common case.**

Example:

```text
Input:
5 × 5

Filter:
3 × 3

Padding:
same

Output:
5 × 5
```

Keras:

```python
Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding='same'
)
```

Or:

```python
Conv2D(32, 3, padding='same')
```

Keras automatically determines how much padding is needed.

For a `3 × 3` filter, the common `same` case adds:

```text
1 pixel
```

around each side.

---

# Valid vs Same

```text
VALID

No padding is added.

5 × 5
  ↓
3 × 3 filter
  ↓
3 × 3
```

```text
SAME

Padding is automatically added.

5 × 5
  ↓
Padding
  ↓
3 × 3 filter
  ↓
5 × 5
```

---

# Padding With Grayscale Images

A grayscale image can have the shape:

```text
28 × 28 × 1
```

If padding of 1 pixel is added:

```text
30 × 30 × 1
```

The `1` represents the grayscale channel and does not change.

---

# Padding With RGB Images

An RGB image can have the shape:

```text
28 × 28 × 3
```

After adding 1-pixel padding:

```text
30 × 30 × 3
```

Padding changes the:

```text
Height
Width
```

It does not add extra RGB channels.

---

# Padding Does Not Create New Images

Suppose you have:

```text
60,000 images
```

Each image is:

```text
28 × 28
```

Adding padding changes the size of each image:

```text
28 × 28
   ↓
30 × 30
```

But the number of images remains:

```text
60,000
```

Padding only adds a border around each image.


---

# ⭐ Important Points

◇ **Padding = adding pixels around the border of an image.**

◇ The most common padding is **zero padding**.

◇ `padding='valid'` → **No padding**.

◇ `padding='same'` → **Keras automatically adds the required padding**.

◇ Padding helps the CNN process **edge pixels**.

◇ Padding can prevent the feature map from becoming smaller.

◇ Padding changes the **height and width** of the image.

◇ Padding does **not** create additional images.

◇ Padding does **not** create additional color channels.

### One-line definition

> **Padding is the process of adding extra pixels, usually zeros, around the border of an image before convolution.**
