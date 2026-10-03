# Integer Encoding 

##  What Is Integer Encoding?

**Integer encoding** is a method of converting text tokens into unique integer IDs.

Neural networks work with numbers, so we first need to represent words/tokens using numbers.

Example:

```text
"I love you"
```

After integer encoding:

```text
[1, 2, 3]
```

Here:

```text
I     → 1
love  → 2
you   → 3
```

The numbers are **IDs** for the tokens.

---

# 2. Why Do We Need Integer Encoding?

A neural network cannot directly process:

```text
"I love you"
```

as ordinary text.

We first convert the text into tokens and then assign each token an integer.

```text
Text
 ↓
Tokens
 ↓
Integer IDs
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
```

---

# 3. Tokenization

Before integer encoding, we usually perform **tokenization**.

## Definition

**Tokenization is the process of splitting text into smaller units called tokens.**

For simple word-level tokenization:

```text
"I love you"
```

becomes:

```text
["I", "love", "you"]
```

Another example:

```text
"I like you so much"
```

becomes:

```text
["I", "like", "you", "so", "much"]
```

The tokens can be words, subwords, or characters depending on the tokenizer.

---

# 4. Vocabulary

## Definition

A **vocabulary** is the collection of *unique* tokens known by the tokenizer.

For our example:

```text
I
love
you
like
so
much
```

Therefore:

```text
Vocabulary size = 6
```

We can assign an integer ID to every token:

```text
I     → 1
love  → 2
you   → 3
like  → 4
so    → 5
much  → 6
```

---

# 5. Token ID

## Definition

A **token ID** is the integer assigned to a particular token.

For example:

```text
I → 1
```

Here:

```text
1 = token ID of "I"
```

Similarly:

```text
love → 2
you  → 3
```

The ID is mainly an identifier.

---

# 6. Example: Integer Encoding

Suppose our sentences are:

```text
I love you

I like you so much
```

Our vocabulary is:

```text
I     → 1
love  → 2
you   → 3
like  → 4
so    → 5
much  → 6
```

Now encode the sentences.

### Sentence 1

```text
"I love you"

"I"    → 1
"love" → 2
"you"  → 3

Result:

[1, 2, 3]
```

### Sentence 2

```text
"I like you so much"

"I"    → 1
"like" → 4
"you"  → 3
"so"   → 5
"much" → 6

Result:

[1, 4, 3, 5, 6]
```

Therefore:

```text
"I love you"
→ [1, 2, 3]

"I like you so much"
→ [1, 4, 3, 5, 6]
```

---

# 7. Vocabulary Size vs Token ID

These are different things.

## Vocabulary Size

If:

```text
Vocabulary size = 100
```

it means:

```text
There are 100 unique tokens.
```

## Token ID

A particular token might have:

```text
"I" → 1
```

So:

```text
Vocabulary size = 100
Token ID of "I" = 1
```

Do not confuse the two.

---


# 9. Important: Integer IDs Are Just Identifiers

Suppose:

```text
cat → 1
dog → 2
car → 3
```

It does **not** mean:

```text
dog is mathematically closer to cat
```

just because:

```text
2 is closer to 1 than 3 is.
```

The integer itself is not a measurement of meaning.

Think of it like an ID card:

```text
Person A → ID 101
Person B → ID 102
Person C → ID 103
```

ID `102` does not mean Person B is "more similar" to Person A than Person C.

The same idea applies to token IDs.

---

# 10. Problem of Different Sentence Lengths

Consider:

```text
"I love you"
→ [1, 2, 3]
```

and:

```text
"I like you so much"
→ [1, 4, 3, 5, 6]
```

Their lengths are different:

```text
[1, 2, 3]       → length 3

[1, 4, 3, 5, 6] → length 5
```

When processing multiple sequences together, it is often useful to make them the same length.

This is where **padding** is used.

---

#  Padding

## Definition

**Padding adds extra values, usually `0`, so sequences have the same length.**

Our longest sequence has length 5:

```text
[1, 4, 3, 5, 6]
```

So we can pad the shorter sequence to length 5.

Before padding:

```text
[1, 2, 3]
```

After padding:

```text
[1, 2, 3, 0, 0]
```

Now both sequences have length 5:

```text
[1, 2, 3, 0, 0]

[1, 4, 3, 5, 6]
```

---

#  Why Do We Use Padding?

Neural networks often process multiple sequences together as a batch.

For efficient batch processing, the sequences are commonly represented with the same shape.

For example:

```text
Sequence 1 → [1, 2, 3, 0, 0]
Sequence 2 → [1, 4, 3, 5, 6]
```

Both now have:

```text
Sequence length = 5
```

This creates a rectangular numerical structure.

---

#  `maxlen`

In Keras, you may encounter:

```python
maxlen = 5
```

This means the output sequence length is limited to 5.

For example:

```text
[1, 2, 3]
```

can become:

```text
[1, 2, 3, 0, 0]
```

If the original sequence is longer than 5, it can be truncated depending on the padding/truncation settings.

So:

```text
maxlen = maximum sequence length
```

---

# 14. Padding and Truncation

Suppose:

```text
maxlen = 5
```

### Short sequence

```text
[1, 2, 3]

↓ padding

[1, 2, 3, 0, 0]
```

### Long sequence

```text
[1, 2, 3, 4, 5, 6, 7]

↓ truncation to maxlen 5

[1, 2, 3, 4, 5]
```

Exactly which values are removed depends on the truncation configuration.

---

# 16. Keras Example

A modern Keras workflow can use `TextVectorization` to perform tokenization and integer encoding.

```python
import tensorflow as tf

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=5
)
```

## What does this do?

```python
max_tokens=10000
```

means the vocabulary(unique words /tokens) is limited to at most 10,000 tokens.optional()

```python
output_mode="int"
```

means the layer outputs integer token IDs.

```python
output_sequence_length=5
```

means every output sequence has length 5.

It handles padding/truncation to that requested length.

---



# What Integer Encoding Gives Us

Integer encoding gives us:

```text
Text
 ↓
Numbers
```

For example:

```text
"I love you"
→ [1, 2, 3]
```

This is useful because the neural network can now receive numerical input.

However, the numbers are still primarily **token IDs**.

They do not automatically contain meaningful semantic relationships.

---

#Integer Encoding Is the First Numerical Representation

It is useful to remember:

```text
Text
 ↓
Tokenization
 ↓
Integer Encoding
 ↓
Integer IDs
```

For example:

```text
"I love you"
 ↓
["I", "love", "you"]
 ↓
[1, 2, 3]
```

Later, an embedding layer can use these IDs to produce learned vectors.

That is a separate step.

---

#  Final Mental Model

Think of integer encoding as assigning an ID to every token.

```text
"I"     → ID 1
"love"  → ID 2
"you"   → ID 3
```

Then a sentence becomes:

```text
"I love you"
→ [1, 2, 3]
```

If we need equal-length sequences:

```text
[1, 2, 3]
→ [1, 2, 3, 0, 0]
```

So:

```text
Tokenization
= Split text into tokens

Vocabulary
= Collection of unique tokens

Token ID
= Integer assigned to a token

Integer Encoding
= Convert tokens into integer IDs

Padding
= Make sequences the same length

Truncation
= Shorten sequences that exceed the maximum length
```

## One-Line Definition

> **Integer encoding converts tokenized text into integer IDs so that text can be represented numerically and passed into later NLP processing steps.**
