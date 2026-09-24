# Data Augmentation

Data Augmentation is a technique used to create different variations of training images by applying random transformations.

It helps the CNN **reduce overfitting** and **generalize better**.

## Common Techniques

```python
keras.layers.RandomFlip("horizontal")
keras.layers.RandomRotation(0.2)
keras.layers.RandomZoom(0.2)
keras.layers.RandomTranslation(0.1, 0.1)
keras.layers.RandomBrightness(0.2)
keras.layers.RandomContrast(0.2)
```

## On-the-Fly Augmentation

Augmentation can be added directly to the model:

```python
model.add(keras.layers.RandomFlip("horizontal"))
model.add(keras.layers.RandomRotation(0.2))
model.add(keras.layers.RandomZoom(0.2))
```

The images are augmented **during training** and are not saved as new files.

```text
1000 original images
        ↓
Random augmentation
        ↓
Different variations during training
```

## Saving Augmented Images

If we generate and save augmented images:

```python
for i in range(10):
    augmented = augmentation(image, training=True)
    keras.utils.save_img(f"aug_{i+1}.jpg", augmented[0])
```

For example:

```text
1000 images × 10 augmentations
= 10,000 augmented images
```

If the original images are also kept:

```text
1000 original + 10,000 augmented
= 11,000 total files
```

## Important

Augmentation should normally be applied **only to the training set**.

```text
Training   → Augmentation ✅
Validation → No augmentation ❌
Test       → No augmentation ❌
```

## Key Point

> Data augmentation increases the variety of training examples through transformations such as flipping, rotation, zooming, and translation, helping the model reduce overfitting and generalize better.
