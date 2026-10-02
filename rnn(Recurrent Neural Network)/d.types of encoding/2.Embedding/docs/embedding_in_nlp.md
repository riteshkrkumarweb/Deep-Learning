# Embedding in NLP

##  What Is an Embedding?
An **embedding** is a numerical representation of a token as a vector of multiple values.

We already converted text into integer IDs:

```text
"I love you"
     ↓
[1, 2, 3]
```

The embedding layer takes those IDs(integers) and maps them to vectors:

```text
1 → [0.20, 0.30]
2 → [0.80, 0.70]
3 → [0.25, 0.35]
```

So:

```text
Integer ID
    ↓
Embedding
    ↓
Vector
```

---

#  Why Do We Need Embedding?

Integer encoding gives us token IDs.

For example:

```text
I     → 1
love  → 2
you   → 3
```

The problem is that these numbers are mainly **identifiers**.

The model should not assume:

```text
love = 2
like = 3
```

means that `love` and `like` are similar just because `2` and `3` are close numbers.

Similarly:

```text
cat → 1
dog → 2
car → 3
```

does not mean:

```text
cat and dog are similar
```

because:

```text
1 and 2 are close
```

Integer IDs do not contain semantic meaning by themselves.

Embedding provides a richer numerical representation.

---

#  Integer Encoding → Embedding

The typical flow is:

```text
Text
 ↓
Tokenization
 ↓
Integer Encoding
 ↓
Integer IDs
 ↓
Embedding Layer
 ↓
Vectors
 ↓
Neural Network
```

Example:

```text
"I love you"
     ↓
["I", "love", "you"]
     ↓
[1, 2, 3]
     ↓
Embedding
     ↓
[
  [0.20, 0.30],
  [0.80, 0.70],
  [0.25, 0.35]
]
```

The embedding layer does not normally take raw words directly.

It receives the token IDs.

---

# What Is a Vector?

A **vector** is an ordered collection of numbers.

Example:

```text
[0.20, 0.30]
```

This vector contains 2 numbers.

Another example:

```text
[0.20, 0.30, 0.50, 0.80]
```

contains 4 numbers.

For an embedding:

```text
love → [0.80, 0.70]
```

the vector is the numerical representation of that token.

---

#  Embedding Dimension

The number of values in an embedding vector is called the **embedding dimension**.

Examples:

### 1-dimensional

```text
love → [0.80]
```

### 2-dimensional

```text
love → [0.80, 0.70]
```

### 4-dimensional

```text
love → [0.80, 0.70, 0.20, 0.40]
```

### 128-dimensional

```text
love → [128 numbers]
```

Therefore:

```text
output_dim = embedding dimension
```

---

#  Why Use Multiple Numbers?

An embedding does not have to contain 4 numbers.

We use multiple dimensions because a single number provides very limited capacity for representing complex patterns.

With multiple dimensions, the model has a larger numerical space in which it can learn useful relationships.

For example:

```text
2D:

love → [0.80, 0.70]
like → [0.75, 0.65]
car  → [0.10, 0.90]
```

The model can learn representations in this space.

Important:

> The dimensions do not normally correspond to simple human labels(understand) such as "love-ness" or "emotion."

The neural network learns useful numerical patterns automatically.

---



#  Embedding Layer as a Lookup Table

A useful way to understand an embedding layer is as a **learnable lookup table**.

Suppose the vocabulary has 6 tokens and the embedding dimension is 2:

```text
Token ID     Vector
────────────────────────
1            [0.20, 0.30]
2            [0.80, 0.70]
3            [0.25, 0.35]
4            [0.75, 0.65]
5            [0.10, 0.80]
6            [0.40, 0.90]
```

If the input is:

```text
[1, 4, 3]
```

the embedding layer retrieves:

```text
[
  [0.20, 0.30],
  [0.75, 0.65],
  [0.25, 0.35]
]
```

So:

```text
ID 1 → vector 1
ID 4 → vector 4
ID 3 → vector 3
```

---

#  Embedding Is Learnable

The important difference from simple integer encoding is that embedding values can be **learned during neural-network training**.

For example, initially an embedding might contain arbitrary values:

```text
love → [0.12, -0.71]
```

During training, the values can be updated.

Later they might become:

```text
love → [0.80, 0.70]
```

The exact values are determined by training.

The example values in these notes are illustrative.

---

#  How Does the Neural Network Learn the Embedding?

The neural network does not start with the knowledge:

```text
"I" and "you" are related.
```

Instead, it learns from training examples.

A simplified process is:

```text
Training Text
      ↓
Token IDs
      ↓
Embedding
      ↓
Neural Network
      ↓
Prediction
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Gradient Descent
      ↓
Update Model Parameters
      ↓
Embedding Values Change
```

This process is repeated many times.

---

#  Example of Learning From Context

Suppose training data contains:

```text
I love you
I like you
You love me
You like me
```

The model repeatedly sees tokens in different contexts.

The training process adjusts the model parameters to reduce prediction error.

The embedding values are part of those parameters.

Over time, tokens that occur in related contexts can develop related representations.

This is why embeddings can capture useful relationships.

---

#  The Neural Network Does Not Use a Manual Rule

The model is not given a rule like:

```text
if word == "I":
    make it similar to "you"
```

Instead:

```text
Training examples
       ↓
Predictions
       ↓
Loss
       ↓
Backpropagation
       ↓
Parameter updates
       ↓
Learned embeddings
```

The relationships emerge from the training process.

---

#  Similar Vectors

Suppose the model learns:

```text
I   → [0.20, 0.30]
you → [0.25, 0.35]
car → [0.90, 0.10]
```

In this illustrative example, `I` and `you` are relatively close.

This can indicate that the model has learned related contextual or linguistic information about them.

However:

```text
Similar vector
≠
Same word
≠
Synonym
```

Words can have related representations without having the same meaning.

---

#  Measuring Similarity

A common way to compare embedding vectors is **cosine similarity**.

Cosine similarity measures the angle/direction between vectors.

Conceptually:

```text
Cosine similarity ≈ 1
→ very similar direction

Cosine similarity ≈ 0
→ little directional similarity

Cosine similarity ≈ -1
→ opposite direction
```

For example:

```text
love → [0.80, 0.70]
like → [0.75, 0.65]
```

These may have high cosine similarity if their directions are close.

---

#  Important: Similarity Depends on Training

There is no universal guarantee that:

```text
I
```

and:

```text
you
```

will always have similar vectors.

The learned representations depend on:

```text
Training data
Model architecture
Training objective
Initialization
Optimization
Amount of training
```

Therefore, the vectors in examples are only illustrations.

---

# 17. Keras Embedding Layer

A simple Keras embedding layer is:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding

model = Sequential([
    Embedding(input_dim=100, output_dim=2)
])
```

---

#  Understanding `input_dim`

```python
input_dim=100
```

means the embedding layer has 100 possible token indices.

In a simplified example:

```text
Vocabulary size = 100
```

So the embedding table has a row for each possible token index.

Conceptually:

```text
100 token IDs
      ↓
100 embedding vectors
```

---

#  Understanding `output_dim`(in how many spaces u want want to visulaize(hyperparamter for rnn))

```python
output_dim=2
```

means each token gets a vector containing 2 values.

For example:

```text
ID 1 → [0.20, 0.30]
ID 2 → [0.80, 0.70]
ID 3 → [0.25, 0.35]
```

Therefore:

```text
input_dim
→ Number of possible token IDs

output_dim
→ Number of values in each embedding vector
```

---

# Shape of the Embedding Output

Suppose:

```text
Input:
[1, 2, 3]
```

and:

```python
Embedding(input_dim=100, output_dim=2)
```

Then the output contains one 2-dimensional vector for each input token.

Conceptually:

```text
Input shape:
(3,)

Output shape:
(3, 2)
```

Meaning:

```text
3 tokens
×
2 embedding values per token
```

If we have a batch of 32 sequences, each with length 5:

```text
Input shape:
(32, 5)
```

Then:

```text
Embedding output shape:
(32, 5, 2)
```

Meaning:

```text
32 sequences
×
5 tokens per sequence
×
2 values per token
```

---

#  Complete Example

Suppose integer encoding gives:

```text
"I love you"
→ [1, 2, 3]
```

We pass it through:

```python
Embedding(input_dim=100, output_dim=2)
```

The embedding layer might produce:

```text
1 → [0.20, 0.30]
2 → [0.80, 0.70]
3 → [0.25, 0.35]
```

Therefore:

```text
[1, 2, 3]
       ↓
[
  [0.20, 0.30],
  [0.80, 0.70],
  [0.25, 0.35]
]
```

The exact values are learned during training.

---

#  Complete NLP Flow

The complete pipeline is:

```text
Raw Text
   ↓
Tokenization
   ↓
Tokens
   ↓
Integer Encoding
   ↓
Integer IDs
   ↓
Padding / Truncation
   ↓
Fixed-Length Integer Sequences
   ↓
Embedding
   ↓
Embedding Vectors
   ↓
RNN / LSTM / GRU / Transformer
   ↓
Prediction
```

Example:

```text
"I love you"
      ↓
["I", "love", "you"]
      ↓
[1, 2, 3]
      ↓
[1, 2, 3, 0, 0]
      ↓
Embedding
      ↓
[
  [0.20, 0.30],
  [0.80, 0.70],
  [0.25, 0.35],
  [0.00, 0.00],
  [0.00, 0.00]
]
      ↓
Neural Network
```

---

#  Integer Encoding vs Embedding

| Feature | Integer Encoding | Embedding |
|---|---|---|
| Output | Integer ID | Vector |
| Example | `love → 2` | `love → [0.8, 0.7]` |
| Values per token | 1 | Multiple |
| Main purpose | Identify token | Represent token numerically |
| Semantic relationships | Not encoded by ID itself | Can be learned |
| Learnable values | IDs are assigned | Vector values are learned |
| Example output | `[1, 2, 3]` | `[[...], [...], [...]]` |

---

#  Why Integer Encoding Comes First

The embedding layer normally receives **integer token IDs**, not raw words.

Therefore:

```text
Text
 ↓
Tokenization
 ↓
Integer Encoding
 ↓
Token IDs
 ↓
Embedding
 ↓
Vectors
```

For example:

```text
"I love you"
      ↓
["I", "love", "you"]
      ↓
[1, 2, 3]
      ↓
Embedding
      ↓
[
 [0.20, 0.30],
 [0.80, 0.70],
 [0.25, 0.35]
]
```

So integer encoding and embedding are connected, but they are not the same thing.

---

#  Key Terms

### Token

A unit of text processed by the tokenizer.

```text
"I love you"
→ "I", "love", "you"
```

### Token ID

The integer assigned to a token.

```text
love → 2
```

### Embedding

A learned vector representation of a token.

```text
love → [0.80, 0.70]
```

### Embedding Dimension

The number of values in each embedding vector.

```text
[0.80, 0.70]
        ↑
dimension = 2
```

### Embedding Layer

A neural-network layer that maps token IDs to embedding vectors.

---

#  Final Mental Model

Remember:

```text
TOKEN
  ↓
ID
  ↓
VECTOR
```

Example:

```text
love
 ↓
2
 ↓
[0.80, 0.70]
```

The three things are different:

```text
"love"
   ↑
actual token

2
 ↑
token ID

[0.80, 0.70]
 ↑
embedding vector
```

## One-Line Definition

> **An embedding converts integer token IDs into learned vectors of multiple numerical values so that a neural network can work with useful representations of text.**
