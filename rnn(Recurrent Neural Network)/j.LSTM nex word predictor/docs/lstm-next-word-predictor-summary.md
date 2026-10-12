# LSTM Next Word Predictor — Summary Notes

## 1. What Is a Next Word Predictor?

A **next word predictor** predicts which word is likely to come next based on the text already provided.

### Practical example

Input:

`I am learning machine`

Possible next word:

`learning`

The actual prediction depends on the patterns learned from the training dataset.

### Real-world applications

- Smartphone keyboards that suggest the next word.
- Gmail Smart Compose, which suggests text as you write.
- Code completion tools that predict the next part of code.

The tutorial uses an **LSTM (Long Short-Term Memory)** network to build a next-word predictor.

---

## 2. Convert Text Generation into Supervised Learning

Text generation can be framed as a supervised learning problem by creating input-output pairs from existing text.

Consider this sentence:

`Hi my name is Bittu`

The training examples become:

| Input (X) | Target (y) |
|---|---|
| `Hi` | `my` |
| `Hi my` | `name` |
| `Hi my name` | `is` |
| `Hi my name is` | `Bittu` |

The model learns to associate each input sequence with its next word.

### Workflow

1. Start with existing text.
2. Create input sequences and their next-word targets.
3. Convert words into numbers.
4. Train the LSTM on these examples.
5. Give the trained model new text and ask it to predict the next word.

**Key idea:** Each training target is the next word following its input sequence.

---

## 3. Why Convert Text into Numbers?

Deep learning models operate on numerical representations rather than raw words. Therefore, words are converted into integer IDs before being passed to the LSTM.

Suppose a tokenizer assigns these IDs:

| Word | Token ID |
|---|---:|
| `Hi` | 1 |
| `my` | 2 |
| `name` | 3 |
| `is` | 4 |
| `Bittu` | 5 |

The sentence:

`Hi my name is Bittu`

becomes:

`[1, 2, 3, 4, 5]`

The input-output pairs can now be represented numerically:

| Input sequence (X) | Target (y) |
|---|---:|
| `[1]` | 2 |
| `[1, 2]` | 3 |
| `[1, 2, 3]` | 4 |
| `[1, 2, 3, 4]` | 5 |

These IDs are labels for words, not measurements. For example, ID `5` does not mean a word is more important than a word assigned ID `2`.

---

## 4. Using the Keras Tokenizer

The tutorial introduces Keras's `Tokenizer` class to assign IDs to words and convert text into integer sequences.

### Step 1: Learn the vocabulary

The `fit_on_texts()` method analyzes the supplied text and builds a word-to-index mapping.

Example text:

`I love machine learning`

An illustrative mapping might be:

```python
{
    "i": 1,
    "love": 2,
    "machine": 3,
    "learning": 4
}
```

The actual IDs depend on the tokenizer's vocabulary and settings. This example is illustrative, not the actual vocabulary from the tutorial dataset.

### Step 2: Convert text into sequences

The `texts_to_sequences()` method uses the learned mapping to transform text into lists of integers.

Input:

```python
["I love machine learning"]
```

Illustrative output:

```python
[[1, 2, 3, 4]]
```

The outer list represents the collection of input texts; the inner list contains the token IDs for one text.

---

## 5. Why Split the Dataset into Lines?

The tutorial splits the complete text using newline characters so it can process individual lines.

Example:

```text
I live in Gurgaon
I am learning AI
```

This becomes two text units:

- `I live in Gurgaon`
- `I am learning AI`

Each line can then be converted into an integer sequence using `texts_to_sequences()`.

**Note:** Splitting on newline characters creates lines, not necessarily grammatically complete sentences. The tutorial uses lines as the units of processing.

---

## 6. Overall Workflow

1. **Raw text dataset** — begin with text such as `Hi my name is Bittu`.
2. **Split text into lines** — separate the text into units.
3. **Tokenizer** — assign word IDs and convert text into sequences.
4. **Create input-target pairs** — use preceding words as input and the next word as the target.
5. **LSTM model** — train the intended model to learn sequence patterns.
6. **Prediction** — provide text and predict a likely next word.

The supplied transcript explains the motivation, supervised-learning formulation, word indexing, line splitting, and conversion to sequences. It ends during the explanation of `texts_to_sequences()`. The later implementation of padding, the LSTM architecture, training, and prediction is not included in the supplied transcript.

---

## 7. Key Takeaways

- **Next-word prediction is a text-generation task.** The model generates text incrementally by predicting what comes next.
- **Supervised learning makes the task trainable.** Each input sequence is paired with its expected next word.
- **Tokenization converts words into integer IDs.** This represents text numerically.
- **`fit_on_texts()` builds the vocabulary mapping** from the supplied text.
- **`texts_to_sequences()` converts text into integer sequences** using the learned mapping.
- **Dataset size matters.** The presenter uses a small FAQ dataset for demonstration and notes that a better model would generally require more data.
- **LSTM is the intended model.** The supplied transcript introduces it as the network for learning from sequences, but this portion does not explain its internal gates or training process.

---

## 8. Quick Revision Table

| Term | Meaning |
|---|---|
| Next-word predictor | Predicts the next word from existing text |
| Supervised learning | Learns from input-target pairs |
| Input sequence | The words available to the model |
| Target word | The expected next word |
| Tokenizer | Assigns IDs to recognized words |
| Vocabulary | The collection of recognized tokens and their mappings |
| Integer sequence | Text represented as a sequence of token IDs |
| LSTM | A recurrent neural network designed to learn sequential patterns |

## Final Concept to Remember

First, transform text into numerical sequences and create input-target pairs. The model then learns to predict the target word from the preceding words. This is the foundation of the LSTM next-word predictor described in the transcript.
