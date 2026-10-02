# Recurrent Neural Network (RNN)
## Forward Propagation and Architecture — Simple Notes

## 1. What is an RNN?

**RNN (Recurrent Neural Network)** is a neural network designed to work with **sequential data**.

Examples of sequential data:

- Text: `I love this movie`
- Time series: temperature over time
- Speech: audio over time

### Simple definition

> An RNN processes data one step at a time and uses information from the previous step to help process the current step.

The important idea is:

```text
Current Input + Previous Memory → New Memory
```

---

# 2. Example: Sentiment Analysis

Suppose we have these reviews:

| Review | Sentiment |
|---|---:|
| movie was good | 1 |
| movie was bad | 0 |
| movie was not good | 0 |

Here:

```text
1 = Positive
0 = Negative
```

The RNN reads the review and predicts its sentiment.

---

# 3. Vocabulary

A **vocabulary** is the list of unique words that the model knows.

Example:

```text
movie
was
good
bad
not
```

So:

```text
Vocabulary size = 5
```

---

# 4. One-Hot Encoding

A computer cannot directly give the word `movie` to a basic neural network.

So we represent each word using numbers.

Example vocabulary:

```text
movie → [1, 0, 0, 0, 0]
was   → [0, 1, 0, 0, 0]
good  → [0, 0, 1, 0, 0]
bad   → [0, 0, 0, 1, 0]
not   → [0, 0, 0, 0, 1]
```

### Simple definition

> One-hot encoding represents a word using a vector containing one `1` and the remaining values as `0`.

---

# 5. Time Steps

An RNN processes a sequence one step at a time.

For:

```text
movie was good
```

we have:

```text
t = 1 → movie
t = 2 → was
t = 3 → good
```

Therefore:

```text
Number of time steps = 3
```

### Simple definition

> A time step is one position in a sequence processed by the RNN.

---

# 6. Input Features

If our vocabulary has 5 words and we use one-hot encoding, each word has 5 values.

For example:

```text
movie = [1, 0, 0, 0, 0]
```

Therefore:

```text
Number of input features = 5
```

For one review:

```text
3 time steps × 5 features
```

So the input shape is:

```text
(3, 5)
```

---

# 7. Batch Size

Suppose we have 100 reviews.

Instead of processing all 100 at once, we can process them in batches.

For example:

```text
batch_size = 32
```

means:

```text
32 reviews are processed together.
```

The complete RNN input normally has this shape:

```text
(batch_size, time_steps, features)
```

Example:

```text
(32, 3, 5)
```

means:

```text
32    → number of reviews
3     → words/time steps per review
5     → features per word
```

---

# 8. Input at Each Time Step

For our example:

```text
x₁ = movie
x₂ = was
x₃ = good
```

Using one-hot encoding:

```text
x₁ = [1, 0, 0, 0, 0]

x₂ = [0, 1, 0, 0, 0]

x₃ = [0, 0, 1, 0, 0]
```

Here `xₜ` means:

> Input at time step `t`.

---

# 9. Hidden State

The **hidden state** is the RNN's memory.

It carries information from previous time steps to the next time step.

Example:

```text
x₁ → RNN → h₁
             ↓
x₂ → RNN → h₂
             ↓
x₃ → RNN → h₃
```

### Simple definition

> Hidden state is the information the RNN carries forward from previous steps.

---

# 10. Initial Hidden State

Before reading the first word, there is no previous information.

So we start with:

```text
h₀ = [0, 0, 0]
```

if there are 3 hidden units.

Usually, a basic RNN starts with a zero hidden state unless another initial state is provided.

---

# 11. Hidden Units

Suppose our RNN has:

```text
3 hidden units
```

Then each hidden state has 3 values.

For example:

```text
h₁ = [0.2, -0.5, 0.8]
```

So:

```text
hidden state shape = (3,)
```

---

# 12. RNN Forward Propagation

At every time step, the RNN uses:

```text
Current Input
      +
Previous Hidden State
      ↓
     RNN
      ↓
New Hidden State
```

The basic equation is:

```text
hₜ = tanh(Wₓ xₜ + Wₕ hₜ₋₁ + bₕ)
```

This looks complicated, but each part is simple.

---

# 13. Meaning of the RNN Equation

```text
hₜ = tanh(Wₓ xₜ + Wₕ hₜ₋₁ + bₕ)
```

### `hₜ`

Current hidden state.

### `xₜ`

Current input.

### `hₜ₋₁`

Previous hidden state.

### `Wₓ`

Weights connecting:

```text
Input → Hidden layer
```

### `Wₕ`

Weights connecting:

```text
Previous hidden state → Current hidden state
```

### `bₕ`

Hidden-layer bias.

### `tanh`

Activation function.

---

# 14. Why Do We Need the Previous Hidden State?

Consider:

```text
movie was not good
```

When the RNN reaches:

```text
good
```

it should not forget:

```text
not
```

The hidden state carries previous information.

Conceptually:

```text
movie
  ↓
h₁
  ↓
was
  ↓
h₂
  ↓
not
  ↓
h₃
  ↓
good
  ↓
h₄
```

So the RNN can use information from earlier words.

---

# 15. Time Step 1

First word:

```text
x₁ = movie
```

Initial memory:

```text
h₀ = [0, 0, 0]
```

The RNN calculates:

```text
h₁ = tanh(Wₓx₁ + Wₕh₀ + bₕ)
```

Because `h₀` starts at zero, the first hidden state mainly receives information from the first input and the bias.

---

# 16. Time Step 2

Second word:

```text
x₂ = was
```

The RNN now uses:

```text
x₂ + h₁
```

Equation:

```text
h₂ = tanh(Wₓx₂ + Wₕh₁ + bₕ)
```

Now `h₂` contains information influenced by:

```text
movie + was
```

---

# 17. Time Step 3

Third word:

```text
x₃ = good
```

The RNN uses:

```text
x₃ + h₂
```

Equation:

```text
h₃ = tanh(Wₓx₃ + Wₕh₂ + bₕ)
```

Now `h₃` contains information influenced by the sequence:

```text
movie → was → good
```

---

# 18. Why Is It Called Recurrent?

The previous hidden state comes back into the next calculation.

```text
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
```

The hidden state is repeatedly passed forward.

### Simple definition

> Recurrent means the network uses its previous state again in the next step.

---

# 19. RNN Weights

There are mainly three important parameter groups in a simple RNN:

```text
Wₓ → input-to-hidden weights
Wₕ → hidden-to-hidden weights
bₕ → hidden bias
```

There is also an output layer with its own weights and bias.

---

# 20. Input-to-Hidden Weight Matrix

Suppose:

```text
Input features = 5
Hidden units = 3
```

Then:

```text
Wₓ shape = (5, 3)
```

Why?

```text
5 input values
      ↓
3 hidden neurons
```

---

# 21. Hidden-to-Hidden Weight Matrix

Suppose:

```text
Hidden units = 3
```

Then:

```text
Wₕ shape = (3, 3)
```

Why?

The previous 3 hidden values connect to the current 3 hidden units.

---

# 22. Hidden Bias

If there are 3 hidden units:

```text
bₕ shape = (3,)
```

Example:

```text
bₕ = [b₁, b₂, b₃]
```

Each hidden unit has a bias.

---

# 23. Same Weights at Every Time Step

This is very important.

The RNN does NOT create new weights for every word.

The same weights are reused:

```text
t=1 → Wₓ, Wₕ, bₕ
t=2 → Wₓ, Wₕ, bₕ
t=3 → Wₓ, Wₕ, bₕ
```

So:

```text
Same weights
     ↓
movie → RNN → h₁
was   → RNN → h₂
good  → RNN → h₃
```

This allows the model to handle sequences of different lengths.

---

# 24. Activation Function: tanh

A basic RNN commonly uses `tanh` for the hidden state.

The formula is:

```text
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

You do not need to calculate this manually.

The important point is:

```text
tanh output range = -1 to +1
```

Examples:

```text
large positive input → close to +1
0                   → 0
large negative input → close to -1
```

---

# 25. Output Layer

After processing the complete sequence, we can use the final hidden state:

```text
h₃
 ↓
Dense layer
 ↓
Sigmoid
 ↓
Prediction
```

For binary sentiment classification, we usually need one output neuron.

---

# 26. Output Equation

The output before sigmoid is:

```text
z = Wᵧh₃ + bᵧ
```

Then:

```text
ŷ = sigmoid(z)
```

Here:

```text
ŷ = predicted probability
```

---

# 27. Sigmoid Function

The sigmoid formula is:

```text
sigmoid(x) = 1 / (1 + e⁻ˣ)
```

Its output is between:

```text
0 and 1
```

Example:

```text
ŷ = 0.90
```

means the model is producing a value close to positive.

Example:

```text
ŷ = 0.10
```

means the model is producing a value close to negative.

The exact class decision depends on the chosen classification threshold, commonly `0.5`.

---

# 28. Complete Forward Propagation

For:

```text
movie was good
```

the flow is:

```text
x₁ = movie
      ↓
    RNN
      ↓
     h₁
      ↓
x₂ = was
      ↓
    RNN
      ↓
     h₂
      ↓
x₃ = good
      ↓
    RNN
      ↓
     h₃
      ↓
   Dense(1)
      ↓
   Sigmoid
      ↓
     ŷ
```

Mathematically:

```text
h₁ = tanh(Wₓx₁ + Wₕh₀ + bₕ)

h₂ = tanh(Wₓx₂ + Wₕh₁ + bₕ)

h₃ = tanh(Wₓx₃ + Wₕh₂ + bₕ)

ŷ = sigmoid(Wᵧh₃ + bᵧ)
```

---

# 29. Shapes in Our Example

Suppose:

```text
Vocabulary/features = 5
Hidden units = 3
Time steps = 3
Output = 1
```

Then:

```text
xₜ      → (5,)
Wₓ      → (5, 3)
hₜ      → (3,)
Wₕ      → (3, 3)
bₕ      → (3,)
Wᵧ      → (3, 1)
bᵧ      → (1,)
ŷ       → (1,)
```

For one complete review:

```text
Input → (3, 5)
```

For a batch of 32 reviews:

```text
Input → (32, 3, 5)
```

---

# 30. Many-to-One RNN

Our sentiment example is a **many-to-one RNN**.

Why?

Many inputs:

```text
movie
was
good
```

produce one output:

```text
sentiment
```

So:

```text
Many time steps
      ↓
     One output
```

Examples:

```text
Text → Sentiment
Speech → Emotion
Sequence → Class
```

---

# 31. Many-to-Many RNN

In many-to-many problems, we can have an output at every time step.

Example:

```text
Input:  I   am   Ritesh
Output: POS VERB NAME
```

Conceptually:

```text
x₁ → RNN → y₁
x₂ → RNN → y₂
x₃ → RNN → y₃
```

This can be useful for tasks such as sequence labeling.

---

# 32. One-to-Many RNN

One input can produce a sequence of outputs.

Example:

```text
Image → Caption words
```

Conceptually:

```text
Input
  ↓
RNN
 ↓
word₁
 ↓
word₂
 ↓
word₃
```

---

# 33. RNN Architecture

A simple many-to-one RNN looks like:

```text
              h₀
               ↓
x₁ ─────────→ RNN ─────→ h₁
                         ↓
x₂ ─────────→ RNN ─────→ h₂
                         ↓
x₃ ─────────→ RNN ─────→ h₃
                         ↓
                       Dense
                         ↓
                      Sigmoid
                         ↓
                      Output
```

The repeated RNN blocks use the **same parameters**.

---

# 34. Keras Example

A simple Keras model can look like:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(3),
    Dense(1, activation="sigmoid")
])
```

### What does this mean?

```text
SimpleRNN(3)
```

means:

```text
RNN with 3 hidden units
```

and:

```text
Dense(1, activation="sigmoid")
```

means:

```text
1 output neuron
+
sigmoid activation
```

This setup is suitable for a simple binary classification example.

---

# 35. The Most Important Formula

Remember this:

```text
hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + bₕ)
```

In simple words:

```text
Current input
     +
Previous memory
     +
Bias
     ↓
Weighted sum
     ↓
tanh
     ↓
New memory
```

Then:

```text
Final memory
     ↓
Output layer
     ↓
Sigmoid
     ↓
Prediction
```

---

# 36. Quick Revision

| Term | Simple meaning |
|---|---|
| Sequence | Ordered data |
| Time step | One position in the sequence |
| Vocabulary | List of known words |
| One-hot encoding | Word represented using 0s and 1 |
| Input `xₜ` | Current input |
| Hidden state `hₜ` | RNN memory |
| `h₀` | Initial hidden state |
| `Wₓ` | Input → hidden weights |
| `Wₕ` | Previous hidden → current hidden weights |
| `bₕ` | Hidden bias |
| `tanh` | Hidden-state activation |
| Dense | Output layer |
| Sigmoid | Converts output to 0–1 |
| Many-to-one | Many inputs → one output |
| Forward propagation | Passing input through the network to get prediction |

---

# 37. One-Line Memory Trick

Remember RNN like this:

```text
Current Word + Previous Memory
              ↓
             RNN
              ↓
         New Memory
              ↓
          Next Word
```

Or even shorter:

> **RNN = Input + Previous Hidden State → New Hidden State**

