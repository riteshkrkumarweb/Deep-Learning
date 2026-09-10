# Stride in CNN

## What is Stride?

**Stride** is the number of pixels by which a convolution filter moves across an image at each step.

In simple words:

> **Stride tells us how far the filter moves after each convolution operation.**

---

## Stride = 1

With `stride = 1`, the filter moves **1 pixel at a time**.

```text
Filter position 1
       ↓
Move 1 pixel →
       ↓
Filter position 2
       ↓
Move 1 pixel →
       ↓
Filter position 3
```

The filter moves through the image with overlapping regions.

### Example

```text
Input:
5 × 5

Filter:
3 × 3

Stride:
1

Output:
3 × 3
```

---

## Stride = 2

With `stride = 2`, the filter moves **2 pixels at a time**.

```text
Filter position 1
       ↓
Move 2 pixels →
       ↓
Filter position 2
       ↓
Move 2 pixels →
```

The filter skips more positions compared with stride 1.

### Example

```text
Input:
5 × 5

Filter:
3 × 3

Stride:
2

Output:
2 × 2
```

---

# How the Filter Moves

Suppose we have a `5 × 5` image and a `3 × 3` filter.

## Stride = 1

The filter moves one pixel at a time:

```text
Position 1 → Position 2 → Position 3
     ↓
Move down 1 pixel
     ↓
Position 4 → Position 5 → Position 6
     ↓
Move down 1 pixel
     ↓
Position 7 → Position 8 → Position 9
```

This gives a:

```text
3 × 3 feature map
```

## Stride = 2

The filter moves two pixels at a time:

```text
Position 1 → Position 2
     ↓
Move down 2 pixels
     ↓
Position 3 → Position 4
```

This gives a:

```text
2 × 2 feature map
```

---

# Advantages of Stride

◇ **Reduces feature-map size**

A larger stride makes the output feature map smaller.

◇ **Reduces computation**

Because the filter visits fewer positions, fewer convolution operations are performed.

◇ **Reduces memory usage**

Smaller feature maps require less memory.

◇ **Can make the model faster**

Fewer operations can make convolution layers faster.

◇ **Can provide downsampling**

A larger stride can reduce spatial dimensions without using a separate pooling layer.

---

# Disadvantages of Stride

◇ **Loss of spatial information**

A larger stride skips some positions, so some fine details may not be captured.

◇ **Can lose small features**

Small objects or fine patterns may disappear when the feature map is reduced too aggressively.

◇ **Less detailed feature maps**

Because the filter checks fewer locations, the resulting feature map can contain less spatial detail.

◇ **Too large a stride can hurt accuracy**

If important information is removed too early, model performance can decrease.

---

# Is Larger Stride Usually Recommended?

A larger stride is **not automatically better**.

It is useful when we intentionally want to reduce the spatial size of feature maps and computation.

Modern computers and GPUs have much more computational power than older systems, so **reducing computation should not be the only reason for choosing a large stride**.

The choice of stride should depend on the architecture and the task.

```text
Need more spatial detail
        ↓
Use smaller stride
        ↓
More positions are processed
```

```text
Need downsampling / lower computation
        ↓
Use larger stride
        ↓
Fewer positions are processed
```

So, rather than saying that stride is "not recommended," it is more accurate to say:

> **Large strides should be used carefully because they reduce spatial information. Modern hardware makes the computational savings less important in some applications, while preserving useful features can be more important.**

---

# Stride in Keras

In Keras, stride is specified using the `strides` parameter of `Conv2D`.

## Stride = 1

```python
from keras.layers import Conv2D

Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1)
)
```

Here:

```text
strides=(1, 1)
```

means:

```text
1 pixel vertically
1 pixel horizontally
```

---

## Stride = 2

```python
Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(2, 2)
)
```

Here:

```text
strides=(2, 2)
```

means:

```text
2 pixels vertically
2 pixels horizontally
```

---

# Stride and Feature Map Size

A larger stride means the filter visits **fewer positions**.

Therefore:

```text
Smaller stride
      ↓
More filter positions
      ↓
More spatial information
      ↓
Larger feature map
```

```text
Larger stride
      ↓
Fewer filter positions
      ↓
Less spatial information
      ↓
Smaller feature map
```

### Example

```text
Input = 5 × 5
Filter = 3 × 3

Stride 1 → Output = 3 × 3

Stride 2 → Output = 2 × 2
```

---

# Important Points

◇ **Stride = how far the filter moves at each step.**

◇ `stride = 1` → filter moves 1 pixel at a time.

◇ `stride = 2` → filter moves 2 pixels at a time.

◇ A larger stride makes the filter skip more positions.

◇ A larger stride generally produces a smaller feature map.

◇ A larger stride reduces computation and memory usage.

◇ A larger stride can also cause loss of spatial information.

◇ Stride does **not** change the size of the filter.

◇ In Keras, stride is controlled using the `strides` parameter.

### Most Important Keras Syntax

```python
# Stride 1
Conv2D(32, 3, strides=(1, 1))

# Stride 2
Conv2D(32, 3, strides=(2, 2))
```

---

# One-Line Definition

> **Stride is the number of pixels by which the convolution filter moves across the image at each step.**


