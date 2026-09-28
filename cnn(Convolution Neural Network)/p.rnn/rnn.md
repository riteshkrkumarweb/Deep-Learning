# RNN (Recurrent Neural Network)

## 1. What is an RNN?

**RNN (Recurrent Neural Network)** is a type of neural network designed for **sequential data**.

It processes data **one step at a time** and carries information from previous time steps using a **hidden state**.

Example:

```text
"I love machine learning"

I → love → machine → learning
```

Conceptually:

```text
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
           ↓
x₄ → RNN → h₄
```

Here:

- `x₁, x₂, x₃, ...` = inputs at different time steps
- `h₁, h₂, h₃, ...` = hidden states
- The hidden state carries information from previous time steps

---

# 2. Why use RNN instead of ANN?

A normal **ANN (Artificial Neural Network)** is generally designed for fixed-size inputs and does not have a built-in recurrent memory for previous time steps.

RNNs are useful when **order and previous context matter**.

## Problem 1: Text input can have varying length

Consider:

```text
"I love AI"

"I love AI and machine learning"

"AI is changing the way we build software"
```

These sentences contain different numbers of words.

An ANN generally expects a fixed-size input.

RNNs can process the sequence step-by-step:

```text
word₁ → RNN
word₂ → RNN
word₃ → RNN
...
wordₙ → RNN
```

---

## Problem 2: Zero padding

One way to make variable-length sequences compatible with a fixed-size ANN is to pad shorter sequences.

For example:

```text
I → love → AI → PAD → PAD → PAD → PAD
```

The `PAD` values do not contain useful information.

Padding is still used with RNNs in practical systems, usually together with masking or other sequence-handling techniques, so padding itself is not the fundamental reason to use RNNs.

The important advantage is that RNNs are designed to process sequential information.

---

## Problem 3: Previous information matters

Consider:

```text
"The movie was not good"
```

When the model reaches:

```text
good
```

the previous word:

```text
not
```

is important for understanding the sentence.

An ordinary feed-forward ANN does not naturally carry information from one time step to another.

An RNN maintains a hidden state:

```text
not
 ↓
RNN
 ↓
hidden state
 ↓
good
 ↓
RNN
```

Therefore, the current output can depend on information from previous steps.

---

## Problem 4: Sequence order matters

Compare:

```text
"Dog bites man"
```

with:

```text
"Man bites dog"
```

The words are similar, but their order is different.

An RNN processes them in sequence:

```text
Dog → bites → man
```

The hidden state changes as each element is processed.

Therefore, RNNs can model information related to **sequence order**.

---

# 3. ANN vs RNN

| Feature | ANN | RNN |
|---|---|---|
| Designed specifically for sequences | No | Yes |
| Fixed-size input | Usually expected | Can process sequences step-by-step |
| Previous-step memory | No recurrent memory | Hidden state |
| Sequence order | Not naturally modeled | Naturally incorporated |
| Text | Possible with preprocessing | Natural use case |
| Time series | Possible | Natural use case |
| Weight sharing across time | No recurrent sharing | Yes |

---

# 4. How an RNN works

Suppose the input sequence is:

```text
"I love AI"
```

The sequence has three time steps:

```text
x₁ = I
x₂ = love
x₃ = AI
```

The RNN processes them one by one:

```text
I
 ↓
RNN
 ↓
h₁
```

Then:

```text
love + h₁
     ↓
    RNN
     ↓
    h₂
```

Then:

```text
AI + h₂
   ↓
  RNN
   ↓
  h₃
```

The hidden state provides information from previous time steps.

---

# 5. Basic RNN equation

A simplified RNN can be represented as:

```text
hₜ = f(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)
```

Where:

- `xₜ` = current input
- `hₜ₋₁` = previous hidden state
- `hₜ` = current hidden state
- `Wₓₕ` = input-to-hidden weights
- `Wₕₕ` = hidden-to-hidden/recurrent weights
- `b` = bias
- `f` = activation function, commonly `tanh` in a basic RNN

The important idea is:

```text
Current input
      +
Previous memory
      ↓
   RNN cell
      ↓
Current memory
```

---

# 6. RNN and text

Before text can be given to an RNN, it needs to be converted into numbers.

For example:

```text
Text
 ↓
Tokenization
 ↓
Integer token IDs
 ↓
Embedding
 ↓
RNN
 ↓
Output
```

## Vocabulary

Suppose our vocabulary contains 12 words.

```text
Vocabulary size = 12
```

A one-hot representation could therefore contain 12 positions:

```text
word → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Important:

**12 is the vocabulary size in this example, not the number of words in every sentence.**

For example:

```text
"I love AI"
```

contains 3 words/time steps, even if the vocabulary contains 12 words.

---

# 7. What is a time step?

A **time step** is one position in a sequence.

For:

```text
"I love machine learning"
```

we have:

```text
Time step 1 → I
Time step 2 → love
Time step 3 → machine
Time step 4 → learning
```

So:

```text
Number of time steps = 4
```

---

# 8. Important RNN applications

RNNs can be used for:

- Sentiment analysis
- Text classification
- Next-word prediction
- Text generation
- Speech-related sequence tasks
- Time-series prediction
- Sequence classification
- Sequence forecasting

Example:

```text
"I really enjoyed this movie"
          ↓
        RNN
          ↓
      Positive
```

---

# 9. Simple mental model

Remember RNN like this:

```text
        Previous information
                ↓
Current input → RNN → Current information
                ↓
        Next time step
```

Or simply:

```text
Input → Process → Remember → Next input → Process → Remember
```

---

# 10. Key point

The main reason to use an RNN is:

> **RNNs are designed to process sequential data while carrying information from previous time steps through a hidden state.**

So:

```text
ANN
Fixed input → Model → Output

RNN
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
           ↓
x₄ → RNN → h₄
```

**ANN:** mainly treats the input as a fixed collection of features.

**RNN:** processes the sequence step-by-step and maintains a recurrent hidden state.
