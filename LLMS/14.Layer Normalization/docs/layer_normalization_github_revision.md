# Layer Normalization in Deep Learning and Transformers

## 1. Overview

This lesson explains:
- What normalization means in deep learning.
- Why normalization is useful.
- How Batch Normalization works.
- Why Batch Normalization is problematic for variable-length sequential data.
- How Layer Normalization works.
- Why Transformers use Layer Normalization.

The main idea:

greater than  **Batch Normalization normalizes across the batch, while Layer Normalization normalizes across the features of each individual example/token.**

---

## 2. What Is Normalization?

Normalization is a mathematical transformation that adjusts data to have certain statistical properties or to fit within a particular range.

Two familiar examples are:

### Standardization

For a value $x$:

$$
x_{\text{standardized}} = \frac{x-\mu}{\sigma}
$$

Where:
- $x$ = original value
- $\mu$ = mean
- $\sigma$ = standard deviation

After standardization, the transformed values have approximately:
- Mean = 0
- Standard deviation = 1

### Min-Max Scaling

Min-Max scaling transforms values into a chosen range, commonly 0 to 1:

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

---

## 3. Where Is Normalization Used in Deep Learning?

Normalization can be applied to two broad kinds of values:

1. **Input data:** features supplied to the neural network.
2. **Hidden-layer activations:** values produced by neurons inside the network.

This lesson focuses mainly on normalizing hidden-layer activations.

## 4. Why Is Normalization Useful?

The transcript highlights four benefits:

1. **More stable training:** helps reduce extreme activation values that can contribute to unstable gradients.
2. **Faster convergence:** more consistent values can help optimization progress more smoothly.
3. **Helps with internal covariate shift:** hidden-layer inputs can change as earlier layer weights are updated during training.
4. **Possible regularization effect:** some normalization methods, including Batch Normalization, can have a regularizing effect.

### Covariate Shift vs. Internal Covariate Shift

- **Covariate shift:** the input-data distribution changes between training and prediction.
- **Internal covariate shift (as described in the lesson):** the distribution of activations entering a hidden layer changes during training because the weights in earlier layers are being updated.

The lesson uses a flower-classification example: a model trained only on red roses may struggle if it later receives yellow or white roses.

---

## 5. Batch Normalization

### Meaning

Batch Normalization (BatchNorm) normalizes activations by calculating statistics across examples in a mini-batch, separately for each feature/neuron.

Imagine a hidden layer with three neurons. For each input row, the layer produces:

- $z_1$: pre-activation of neuron 1
- $z_2$: pre-activation of neuron 2
- $z_3$: pre-activation of neuron 3

For BatchNorm:
- Collect the $z_1$ values from all examples in the batch and calculate their mean and standard deviation.
- Do the same independently for $z_2$ and $z_3$.
- Normalize each value using the statistics of its neuron.

### Formula

$$
\hat{z} = \frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y = \gamma\hat{z}+\beta
$$

Where:
- $z$ = original activation
- $\mu_B$ = mean across the batch
- $\sigma_B^2$ = variance across the batch
- $\epsilon$ = small value for numerical stability
- $\gamma$ = learnable scale parameter
- $\beta$ = learnable shift parameter
- $y$ = normalized and transformed activation

The transcript describes $\gamma$ and $\beta$ as learnable parameters, commonly initialized to 1 and 0 respectively.

### Practical Example

Suppose a batch contains five examples. A particular neuron produces:

```text
z = [7, 2, 5, 3, 8]
```

BatchNorm calculates one mean and standard deviation from these five values, then uses them to normalize all five values for that neuron.

It repeats this process separately for every neuron/feature.

### Key Point

**BatchNorm compares the same feature across different examples in the batch.**

---

## 6. Why Batch Normalization Is Problematic for Sequential Data

Transformers process sequences of tokens. Different sentences can have different lengths.

To place multiple sentences into one batch, shorter sentences are padded so that every sequence has the same length.

### Example

```text
Sentence 1: [Hi, Nitesh, PAD, PAD]
Sentence 2: [How, are, you, today]
```

Both sequences now have four token positions.

The padding positions are represented by zero-valued vectors in the lesson's simplified example.

### The Problem

Imagine a batch of 32 sentences:
- The longest sentence has 100 tokens.
- Many other sentences have only around 30 tokens.

Padding creates many extra zero-valued positions in shorter sentences.

If BatchNorm calculates statistics across the batch for each embedding feature, these padding zeros can be included in the mean and standard deviation. They can therefore distort the statistics used to normalize real token representations.

**Why this matters:** the calculated mean and standard deviation may no longer represent the actual non-padding token values well.

This is the central reason given in the transcript for avoiding ordinary BatchNorm in the illustrated self-attention setting.

---

## 7. Layer Normalization

### Meaning

Layer Normalization (LayerNorm) normalizes the features of each individual example independently.

In the neural-network example from the lesson:
- BatchNorm calculates statistics down a feature column, across examples.
- LayerNorm calculates statistics across the features in each row, for each example.

### Formula

For one example with feature values $x_1, x_2, \ldots, x_d$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$$
y_i = \gamma_i\hat{x}_i+\beta_i
$$

Where:
- $d$ = number of features
- $\mu$ = mean across the features of this example
- $\sigma^2$ = variance across those features
- $\gamma_i$, $\beta_i$ = learnable scale and shift parameters for feature $i$

### Practical Example

Suppose one token has a three-dimensional contextual embedding:

```text
Token A = [6.5, 2.41, 6.5]
```

LayerNorm calculates the mean and standard deviation using these three values only. It normalizes the three values and then applies the learnable scale and shift parameters.

It repeats this independently for every token.

### Key Point

**LayerNorm compares the features within one example/token.**

---

## 8. How LayerNorm Handles Padding in the Lesson's Example

Consider a padded token whose embedding is:

```text
[0, 0, 0]
```

LayerNorm calculates statistics across those three features for that token only.

Its mean is 0 and its variance is 0. With the usual zero offset after normalization, the normalized result remains zero (the small $\epsilon$ prevents division by zero).

Most importantly, this padded token's zeros do not enter the mean and variance calculations for another token.

Therefore, in the lesson's illustrated setup, padding in one sequence does not distort the feature statistics of other tokens.

---

## 9. Batch Normalization vs. Layer Normalization

| Comparison | Batch Normalization | Layer Normalization |
|---|---|---|
| Statistics calculated across | Examples in a batch | Features of one example |
| Direction in a data matrix | Down each feature column | Across each example's feature row |
| Depends on other examples in the batch? | Yes | No |
| Padding concern in the illustrated sequence setup | Padding can affect batch statistics | Each token is normalized independently |
| Used in the lesson's Transformer explanation | Not preferred | Preferred |

### Easy Way to Remember

- **BatchNorm:** same feature, many examples.
- **LayerNorm:** many features, one example.

---

## 10. Layer Normalization in Self-Attention

The transcript uses two sentences:

```text
Sentence 1: [Hi, Nitesh]
Sentence 2: [How, are, you, today]
```

For batching, the first sentence is padded:

```text
Sentence 1: [Hi, Nitesh, PAD, PAD]
Sentence 2: [How, are, you, today]
```

Assume each token embedding has three dimensions.

The self-attention module processes the batched input and produces contextual embeddings. The output shape in the simplified example is:

```text
(batch_size, sequence_length, embedding_dimension)
(2, 4, 3)
```

This represents:
- 2 sentences
- 4 token positions per sentence
- 3 features per token

LayerNorm works across the three embedding features for each token position independently.

For example:

```text
Token 1: [feature 1, feature 2, feature 3] -greater than  normalize these 3 values
Token 2: [feature 1, feature 2, feature 3] -greater than  normalize these 3 values
Token 3: [feature 1, feature 2, feature 3] -greater than  normalize these 3 values
```

The padded token is handled independently. Its zero values do not affect the mean and variance of the other tokens.

---

## 11. Important Shape Concept

For the example in the transcript:

```text
Input tensor shape:  (2, 4, 3)
Output tensor shape: (2, 4, 3)
```

LayerNorm changes the values, not the tensor's shape.

The three dimensions mean:

| Dimension | Meaning |
|---|---|
| 2 | Batch size: number of sentences |
| 4 | Sequence length: token positions per sentence |
| 3 | Embedding dimension: features per token |

---

## 12. Final Revision

1. Normalization transforms values to have useful statistical properties.
2. In deep learning, normalization can be applied to inputs and hidden-layer activations.
3. BatchNorm calculates statistics across examples in a batch for each feature.
4. Padding in variable-length sequences can introduce many artificial zeros.
5. Those padding values can distort BatchNorm's batch statistics in the illustrated setup.
6. LayerNorm calculates statistics across features within each individual example/token.
7. In the Transformer example, each token's embedding is normalized independently.
8. LayerNorm preserves the input tensor's shape.

### One-Line Exam Answer

**Transformers commonly use Layer Normalization rather than Batch Normalization because LayerNorm normalizes each token representation across its features independently, avoiding the batch-statistics problem caused by padding in variable-length sequences.**

---

## 13. Quick Self-Test

1. What is the difference between BatchNorm and LayerNorm?
2. In BatchNorm, along which direction are statistics calculated?
3. In LayerNorm, which values are used to calculate the mean and standard deviation?
4. Why can padding be a problem for BatchNorm in variable-length sequences?
5. If the input tensor shape is `(2, 4, 3)`, what does each dimension represent?
6. Does LayerNorm change the tensor's shape?

### Answers

1. BatchNorm normalizes across examples; LayerNorm normalizes across features within each example.
2. Across the batch, separately for each feature.
3. The features belonging to one example/token.
4. Padding adds artificial zeros that can distort batch statistics.
5. Two sequences, four token positions, three features per token.
6. No. It changes values, not shape.

---

## Source Note

These notes summarize the supplied `layer_normalization_transcript.txt` transcript. The examples and explanations follow the lesson's simplified teaching setup.
