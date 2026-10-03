# Types of RNN (Recurrent Neural Network)

RNNs can be classified by looking at how many inputs and outputs are involved in a sequence.

The main types are:

1. One-to-One
2. One-to-Many
3. Many-to-One
4. Many-to-Many

---

# 1. One-to-One

## Definition

One input is given to the model, and the model produces one output.

This is basically the normal input-output structure used in many machine learning models. It does not really use the main sequence-processing ability of an RNN.

## Example

**Image Classification**

Input:
- One image

Output:
- One class label

Example:

```text
Image → Neural Network → "Cat"
```

## Architecture

```text
Input              Output

  x₁ ───────────→   y₁
```

There is one input and one output.

---

# 2. One-to-Many

## Definition

One input is given to the model, and the model produces a sequence of multiple outputs.

The single input is used to generate several outputs over time.

## Example

**Image Captioning**

Input:
- One image

Output:
- A sequence of words

Example:

```text
Image
  ↓
RNN
  ↓
"The" → "cat" → "is" → "sleeping"
```

The image is the single input, while the generated words are multiple outputs.

## Architecture

```text
                    ┌──→ y₁ = "The"
                    │
x₁ ─────────────→ RNN ──→ y₂ = "cat"
                    │
                    ├──→ y₃ = "is"
                    │
                    └──→ y₄ = "sleeping"
```

Another way to visualize it through time:

```text
             h₁        h₂        h₃        h₄
             ↓         ↓         ↓         ↓
x₁ ───────→ RNN ───→ RNN ───→ RNN ───→ RNN
             ↓         ↓         ↓         ↓
            y₁        y₂        y₃        y₄
           "The"     "cat"      "is"   "sleeping"
```

The important idea is:

```text
One Input → Multiple Outputs
```

---

# 3. Many-to-One

## Definition

A sequence of multiple inputs is given to the model, and the model produces one output.

The RNN reads the input sequence and uses the information from the sequence to produce a single result.

## Example

**Sentiment Analysis**

Input:

```text
"I love this movie"
```

The sentence contains multiple words:

```text
"I" → "love" → "this" → "movie"
```

Output:

```text
Positive
```

## Architecture

```text
x₁        x₂        x₃        x₄
"I"     "love"     "this"    "movie"
 ↓         ↓         ↓         ↓
RNN ───→ RNN ───→ RNN ───→ RNN
                              ↓
                         y = Positive
```

Through hidden states:

```text
x₁        x₂        x₃        x₄
 ↓         ↓         ↓         ↓
RNN ───→ RNN ───→ RNN ───→ RNN
 ↓         ↓         ↓         ↓
h₁        h₂        h₃        h₄
                              ↓
                              y
                         "Positive"
```

The important idea is:

```text
Multiple Inputs → One Output
```

## Other Examples

- Spam detection from an email
- Emotion classification from a sentence
- Activity classification from a sequence of sensor readings

---

# 4. Many-to-Many

## Definition

Multiple inputs are given to the model, and the model produces multiple outputs.

Many-to-Many RNNs have **two main types**:

1. Same-length Many-to-Many
2. Variable-length Many-to-Many

---

# 4.1 Many-to-Many: Same Length

## Definition

The number of input steps is equal to the number of output steps.

```text
Input length = Output length
```

The model produces one output for each input in the sequence.

## Example

**Part-of-Speech (POS) Tagging**

Input:

```text
I       love       AI
```

Output:

```text
Pronoun   Verb     Noun
```

There are 3 input words and 3 output tags.

```text
Input:   x₁     x₂     x₃
Output:  y₁     y₂     y₃
```

## Architecture

```text
x₁        x₂        x₃
"I"      "love"     "AI"
 ↓         ↓         ↓
RNN ───→ RNN ───→ RNN
 ↓         ↓         ↓
y₁        y₂        y₃
Pronoun   Verb      Noun
```

### Another Example: Named Entity Recognition (NER)

```text
Ritesh    lives     in      India
   ↓        ↓        ↓        ↓
Person     O        O      Location
```

Here:

```text
4 inputs = 4 outputs
```

So:

```text
Input length = Output length
```

---

# 4.2 Many-to-Many: Variable Length

## Definition

The number of input steps does **not have to be equal** to the number of output steps.

```text
Input length ≠ Output length
```

The model receives one sequence and generates another sequence whose length can be different.

This is commonly called a **Sequence-to-Sequence (Seq2Seq)** architecture.

## Example

**Machine Translation**

Input:

```text
I am learning AI
```

Output:

```text
मैं AI सीख रहा हूँ
```

The input sequence and output sequence can contain different numbers of tokens.

For example:

```text
Input:   x₁ → x₂ → x₃ → x₄

Output:  y₁ → y₂ → y₃ → y₄ → y₅
```

So:

```text
Input length ≠ Output length
```

## Architecture

This type commonly uses an **Encoder and Decoder**.

```text
Input Sequence
      ↓
 ┌──────────┐
 │  Encoder │
 └──────────┘
      ↓
   Context
      ↓
 ┌──────────┐
 │  Decoder │
 └──────────┘
      ↓
Output Sequence
```

### RNN View

```text
x₁       x₂       x₃       x₄
 ↓        ↓        ↓        ↓
RNN ───→ RNN ───→ RNN ───→ RNN
                    │
                    ↓
                  Context
                    ↓
                   RNN
                    ↓
             y₁ ─→ y₂ ─→ y₃ ─→ y₄ ─→ y₅
```

The decoder can continue generating outputs until the sequence is complete.

### Examples

* Machine Translation
* Text Summarization
* Question Answering
* Sequence-to-Sequence generation

---

# Difference Between the Two Many-to-Many Types

| Type | Input Length | Output Length | Example |
|---|---:|---:|---|
| Same Length | Same | Same | POS Tagging |
| Variable Length | Can be different | Can be different | Machine Translation |

### Easy way to remember

```text
Many-to-Many (Same Length)

x₁ → x₂ → x₃ → x₄
↓    ↓    ↓    ↓
y₁ → y₂ → y₃ → y₄

Input length = Output length
```

```text
Many-to-Many (Variable Length)

x₁ → x₂ → x₃ → x₄
          ↓
        Encoder
          ↓
        Decoder
          ↓
y₁ → y₂ → y₃ → y₄ → y₅

Input length ≠ Output length
```

---

# Complete Types of RNN

## One-to-One

```text
One Input → One Output
```

Example:

```text
Image → Class
```

## One-to-Many

```text
One Input → Multiple Outputs
```

Example:

```text
Image → Caption
```

## Many-to-One

```text
Multiple Inputs → One Output
```

Example:

```text
Words → Sentiment
```

## Many-to-Many: Same Length

```text
Multiple Inputs → Multiple Outputs
```

```text
Input length = Output length
```

Example:

```text
Words → POS Tags
```

## Many-to-Many: Variable Length

```text
Multiple Inputs → Multiple Outputs
```

```text
Input length ≠ Output length
```

Example:

```text
Sentence → Translated Sentence
```

---

# Final Summary

```text
                 RNN Types
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
 One-to-One    One-to-Many   Many-to-One
       │            │            │
    1 → 1         1 → Many     Many → 1

                    Many-to-Many
                         │
                ┌────────┴────────┐
                ↓                 ↓
           Same Length       Variable Length
                │                 │
              Many → Many      Many → Many
                │                 │
          Input = Output     Input ≠ Output
                │                 │
           POS Tagging      Translation
```

---

# Complete Comparison

| Type | Input | Output | Example |
|---|---|---|---|
| One-to-One | One | One | Image Classification |
| One-to-Many | One | Many | Image Captioning |
| Many-to-One | Many | One | Sentiment Analysis |
| Many-to-Many | Many | Many | POS Tagging |
| Many-to-Many (Seq2Seq) | Many | Many | Machine Translation |

---

# Easy Way to Remember

```text
One-to-One
1 Input  → 1 Output

One-to-Many
1 Input  → Multiple Outputs

Many-to-One
Multiple Inputs → 1 Output

Many-to-Many
Multiple Inputs → Multiple Outputs
```

---

# Important Note

"One", "Many", "Input", and "Output" describe the **sequence relationship**, not necessarily individual neurons.

For example:

```text
"I love AI"
```

can be represented as a sequence of multiple input tokens:

```text
"I" → "love" → "AI"
```

A Many-to-One RNN can process these inputs one after another and finally produce one output:

```text
"I" → "love" → "AI" → RNN → Positive
```

---

# Summary

## One-to-One

```text
One Input → One Output
```

Example:

```text
Image → Class
```

## One-to-Many

```text
One Input → Multiple Outputs
```

Example:

```text
Image → Words
```

## Many-to-One

```text
Multiple Inputs → One Output
```

Example:

```text
Words → Sentiment
```

## Many-to-Many

```text
Multiple Inputs → Multiple Outputs
```

Example:

```text
Words → POS Tags
```

or:

```text
Sentence → Translated Sentence
```
