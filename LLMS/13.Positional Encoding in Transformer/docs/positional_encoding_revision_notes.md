# Positional Encoding 

##  What is Positional Encoding?

Positional encoding gives a Transformer information about the position of each token in a sentence.

A token is a word or part of a word.

Example:

The dog chased the cat.

The order of the words helps us understand the meaning. If we change the order, the meaning can change.

A Transformer uses self-attention to process tokens together. However, self-attention does not automatically know the original order of the tokens.

Positional encoding adds this missing position information.

##  Why Does a Transformer Need It?

Self-attention has two important advantages:

- It creates contextual representations. A word can use information from other words to understand its meaning.
- It processes tokens in parallel instead of processing them one by one.

But there is a problem.

Consider these sentences:

- Bittu killed the lion.
- The lion killed Bittu.

The words are similar, but their order and meaning are different.

Without positional information, self-attention has no direct signal telling it which token came first, second, or third.

A basic RNN processes tokens step by step, so order is built into its processing. A Transformer needs another way to receive this information.

That is why positional encoding is used.

##  First Idea: Add Position Numbers

Give every token a position number.

Example:

| Token | Position |
|---|---:|
| Bittu | 1 |
| killed | 2 |
| the | 3 |
| lion | 4 |

One simple idea is to add the position number to the token representation.

For example, if an embedding has 512 values, we could add one more value containing the position number. The representation would then have 513 values.

However, this approach has three problems.

### Problem 1: Position numbers can become very large

A long document may contain thousands of tokens. The position number keeps increasing.

Very large values can make neural network training less stable.

One possible solution is to divide each position by the total number of tokens.

But this creates another problem:

| Sentence | Position | Normalized value |
|---|---:|---:|
| Thank you | 2 | 2 divided by 2 = 1 |
| Bittu killed the lion | 2 | 2 divided by 4 = 0.5 |

The second token gets a different value depending on the sentence length.

The same position should have a consistent representation across examples.

### Problem 2: Position numbers are discrete

The values are separate numbers:

1, 2, 3, 4, 5, and so on.

The neural network love using continuous values that change smoothly.

### Problem 3: Relative distance is not represented conveniently

Absolute position means the location of a token from the beginning of the sentence.

Relative position means the distance or relationship between two tokens.

For example:

- How many positions apart are two words?
- Is one word immediately after another?
- Is one word five positions away?

Simple position numbers do not provide a convenient smooth representation for learning these relationships.

### What should a better solution provide?

A good positional encoding should have these properties:

- Bounded values: values stay within a fixed range.
- Continuous values: values change smoothly.
- Useful position relationships: the representation should help express relative distances.

This leads to sine and cosine functions.

##  Second Solution : Use the Sine Function

The sine function has three useful properties:

- Its output stays between -1 and 1.
- It is continuous.
- It is periodic, which means its pattern repeats.

We can use the position of a token as the input to the sine function.

For example:

```text
positional value = sin(position)
```

This solves the problems of unbounded and discrete position numbers.

But a new problem appears.

### Problem: Sine values repeat

A sine wave is periodic. Different positions can eventually produce the same sine value.

Therefore, one sine value cannot uniquely represent every possible position.

## Third Solution: Use Sine and Cosine Together

Instead of using only one value, represent each position using two values:

```text
[sin(position), cos(position)]
```

Now each position has a pair of values.

This makes the representation more useful, but one sine and cosine pair is still periodic.

### Add more sine and cosine pairs

We can create more pairs using different frequencies.

For example:

- First pair uses sine and cosine at one frequency.
- Second pair uses sine and cosine at a lower frequency.
- Third pair uses sine and cosine at an even lower frequency.

Each pair adds two values to the positional encoding vector.

Using multiple frequencies creates a richer pattern for representing positions.

The important idea is that one periodic function can repeat, but several functions with different frequencies provide a more distinctive positional representation.

##  The Sinusoidal Positional Encoding Formula

The original Attention Is All You Need paper introduced these formulas:

```text
Even dimensions:
PE = sin(pos / 10000^(2i / d_model))

Odd dimensions:
PE = cos(pos / 10000^(2i / d_model))
```

### Meaning of the terms

- PE means positional encoding.
- pos means the position of the token.
- i means the index of the sine and cosine pair.
- d_model means the embedding dimension.
- Even dimensions use sine.
- Odd dimensions use cosine.
- 10000 is a constant used to control the frequencies.

### How are dimensions assigned?

The values are arranged in pairs:

| Dimension | Function |
|---:|---|
| 0 | Sine |
| 1 | Cosine |
| 2 | Sine |
| 3 | Cosine |
| 4 | Sine |
| 5 | Cosine |

Every two dimensions use one sine and cosine pair.

If the embedding dimension is 6, there are 3 pairs.

##  Why Are Different Frequencies Used?

The formula changes the frequency for each pair.

- Lower dimensions use higher-frequency waves. Their values change quickly as position changes.
- Higher dimensions use lower-frequency waves. Their values change slowly as position changes.

This gives the model positional information at different scales.

Some dimensions help distinguish nearby positions. Other dimensions change more slowly and provide patterns across larger distances.

##  Numerical Example with 6 Dimensions

Suppose the embedding dimension is 6.

This means the positional encoding also has 6 values.

The first token has position 0 because the paper uses zero-based indexing.

For position 0:

```text
sin(0) = 0
cos(0) = 1
```

All three pairs receive these values.

So the positional encoding is:

```text
[0, 1, 0, 1, 0, 1]
```

For position 1, the formula produces approximately:

```text
[0.8415, 0.5403, 0.0464, 0.9989, 0.0022, 1.0000]
```

You do not need to memorize these decimal values.

Remember that each position receives a vector made from sine and cosine values at different frequencies.

Important indexing note:

- First token has position 0.
- Second token has position 1.
- Third token has position 2.
- Fourth token has position 3.

## 9. Why Must Positional Encoding Have the Same Dimension as the Embedding?

Suppose a token embedding has 6 values.

Its positional encoding must also have 6 values.

Example:

```text
Token embedding:
[0.2, 0.5, 0.1, 0.7, 0.3, 0.9]

Positional encoding:
[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

After addition:
[0.2, 1.5, 0.1, 1.7, 0.3, 1.9]
```

The Transformer adds the two vectors element by element.

The result contains both token information and position information.

### Why add instead of concatenate?

If we concatenate two 6-dimensional vectors, the result has 12 dimensions.

If we add them, the result stays 6-dimensional.

Keeping the same dimension avoids increasing the size of the representation and the computation required by later layers.

Remember:

The positional encoding dimension must equal the token embedding dimension so that element-wise addition is possible.

##  Understanding the Positional Encoding Heatmap

Imagine a sentence containing 50 tokens. Each token has a 128-dimensional positional encoding.

The heatmap has:

- 50 rows, one for each token position.
- 128 columns, one for each dimension.
- Each cell represents one positional value.

Important observations:

1. The first token has position 0, so its values alternate between 0 and 1.
2. Lower dimensions change quickly because they use higher frequencies.
3. Higher dimensions change slowly because they use lower frequencies.
4. Different frequencies work together to represent positions at different scales.

##  Binary Encoding Analogy

The video compares positional encoding with binary number representation.

In binary counting:

- The lowest bit changes very frequently.
- The next bit changes less frequently.
- Higher bits change more slowly.

Sinusoidal positional encoding follows a similar multi-scale idea:

- Some dimensions change quickly.
- Other dimensions change slowly.
- Together, they create a structured representation of position.

The important difference is:

Binary encoding uses discrete values such as 0 and 1. Sinusoidal positional encoding uses continuous values produced by sine and cosine functions.

##  How Does It Help Represent Relative Position?

The video explains an interesting property of sinusoidal positional encoding.

Imagine that every position has a vector:

- v10 represents position 10.
- v20 represents position 20.
- v30 represents position 30.

A suitable linear transformation can move a positional vector forward by a fixed number of positions.

For example, a transformation representing a shift of 10 positions can work like this:

```text
Position 10 becomes position 20.
Position 30 becomes position 40.
Position 4 becomes position 14.
```

The same transformation represents the same shift, regardless of the starting position.

This gives the model a way to represent relationships between positions.

### Why are sine and cosine both needed?

Sine and cosine together have a useful mathematical property.

A shift in position can be represented as a rotation of each sine and cosine pair.

This means a fixed shift can be represented by a consistent linear transformation.

The video introduces this idea but does not fully derive all the mathematics.

##  Complete Flow: Text to Transformer Input

```text
Input sentence
      |
      v
Tokenization
      |
      v
Token IDs
      |
      v
Token embeddings

For every token:
      |
      v
Get its position
      |
      v
Calculate sinusoidal positional encoding
      |
      v
Add positional encoding to token embedding
      |
      v
Position-aware representations
      |
      v
Self-attention
```

##  Important Terms

| Term | Simple meaning |
|---|---|
| Token | A word or part of a word processed by the model |
| Token embedding | A numerical vector representing a token |
| Contextual embedding | A representation influenced by surrounding tokens |
| Positional encoding | Information about a token's position |
| Absolute position | Position measured from the beginning |
| Relative position | Distance or relationship between two positions |
| Bounded | Values stay within a fixed range |
| Continuous | Values change smoothly |
| Periodic | A pattern repeats |
| Frequency | How quickly a periodic pattern changes |
| Dimension | One numerical component of a vector |
| d_model | The embedding dimension |

## 15. Quick Revision

- Self-attention processes tokens in parallel.
- Self-attention does not automatically know the original token order.
- Word order can change sentence meaning.
- Positional encoding provides position information.
- Simple position numbers can become very large.
- Normalizing by sentence length makes the same position receive different values in different-length sentences.
- Sine is bounded, continuous, and periodic.
- A single sine value can repeat at different positions.
- Sine and cosine pairs at different frequencies create a richer positional representation.
- Positional encoding has the same dimension as the token embedding.
- The Transformer adds the two vectors element by element.
- Lower dimensions change faster; higher dimensions change more slowly.
- Sine and cosine allow fixed position shifts to be represented using linear transformations.


## One-Line Summary
Positional encoding adds a same-sized vector of sine and cosine values to each token embedding so that a Transformer receives information about token order and relationships between positions.

