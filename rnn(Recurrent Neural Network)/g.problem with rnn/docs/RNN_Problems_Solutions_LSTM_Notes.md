# Recurrent Neural Network (RNN)

## 1. What is an RNN?

**RNN (Recurrent Neural Network)** is a type of neural network designed for **sequential data**, where the order of the data matters and information from previous steps can help the model understand the current step.

A simple way to understand it:

> **RNN reads a sequence step by step and carries information from previous steps to the next step.**

This makes RNNs useful for data such as:

- Text
- Speech
- Time-series data
- Sensor data
- Stock-price sequences
- Weather measurements

### Why is RNN suitable for text?

Consider:

> **"The movie was not good."**

When the model reaches the word **"good"**, the earlier word **"not"** is important for understanding the complete meaning.

The meaning of:

> "good"

is different from:

> "not good"

So, in sequential data, **previous context can help understand the current information and the overall sequence.**

---

# 2. Basic Idea of RNN

Suppose the sequence is:

> I → love → this → movie

The RNN processes the words one at a time.

```text
"I"       → RNN → h₁
             ↓
"love"    → RNN → h₂
             ↓
"this"    → RNN → h₃
             ↓
"movie"   → RNN → h₄
```

The hidden state carries information from previous steps.

A simplified idea is:

```text
Current input + Previous hidden state
                ↓
              RNN
                ↓
        New hidden state
```

Mathematically:

```text
hₜ = f(Wₓxₜ + Wₕhₜ₋₁ + b)
```

Where:

- `xₜ` = current input
- `hₜ₋₁` = previous hidden state
- `hₜ` = current hidden state
- `Wₓ` = weight for the current input
- `Wₕ` = weight for previous hidden information
- `b` = bias
- `f` = activation function

The important idea is:

> **The current hidden state depends on both the current input and information carried from the previous step.**

---

# 3. The Two Major Problems of Basic RNNs

Basic RNNs have two important training problems:

## Problem A — Long-Term Dependency Problem

Also called the:

> **Vanishing Gradient Problem**

## Problem B — Unstable Training

Also called the:

> **Exploding Gradient Problem**

These problems are related to how gradients behave during **Backpropagation Through Time (BPTT)**.

---

# 4. Problem A — Long-Term Dependency / Vanishing Gradient

## Definition

The **long-term dependency problem** occurs when an RNN has difficulty remembering information from a much earlier part of a sequence.

One major reason is the **vanishing gradient problem**.

> **Vanishing gradient happens when gradients become extremely small while being propagated backward through many time steps, making it difficult for the network to learn long-term relationships.**

---

## Simple Real Example

Imagine this sentence:

> **"I grew up in India. I studied there for many years. ... [many words] ... I speak Hindi."**

To understand the sentence properly, information from **"India"** may be useful much later.

But a basic RNN processes the sentence step by step:

```text
India
  ↓
word 2
  ↓
word 3
  ↓
word 4
  ↓
...
  ↓
word 50
  ↓
Hindi
```

During training, the gradient has to travel backward through many time steps.

If the gradient becomes smaller at every step:

```text
1.0
 ↓
0.5
 ↓
0.25
 ↓
0.125
 ↓
0.0625
 ↓
...
 ↓
almost 0
```

Eventually, the gradient can become so small that the early part of the sequence receives almost no useful learning signal.

Therefore:

> **The RNN may remember recent information but struggle to learn information from far back in the sequence.**

---

# 5. Why Does the Gradient Vanish?

During BPTT, gradients are repeatedly multiplied through many layers/time steps.

If the repeated factors are smaller than 1, their product can become extremely small.

For example:

```text
0.5 × 0.5 × 0.5 × 0.5 × 0.5
= 0.03125
```

With many more multiplications:

```text
0.5 × 0.5 × 0.5 × ... 
≈ 0
```

This is why the gradient can disappear.

### Important idea

The problem is not simply:

> "RNN cannot remember."

The deeper issue is:

> **During training, the gradient may become too small to effectively teach the RNN how to preserve information over long distances.**

---

# 6. Solutions for the Vanishing Gradient Problem

Several techniques can help.

## 6.1 Better Activation Functions

Some activation functions can help gradient flow better than traditional saturating activations in certain neural-network settings.

For example:

- ReLU
- Leaky ReLU
- GELU

However, simply changing the activation function does **not completely solve the fundamental long-term-memory problem of a basic RNN**.

For recurrent networks, the choice must also consider stability and the recurrent architecture.

---

## 6.2 Better Weight Initialization

Good weight initialization can help prevent gradients from becoming badly scaled at the beginning of training.

Examples include:

- Xavier/Glorot initialization
- He initialization

The goal is to start the network with weights that make signal and gradient propagation more stable.

However:

> **Weight initialization helps training, but it does not completely solve long-term dependency in a basic RNN.**

---

## 6.3 Skip/Residual Connections

Skip connections provide a shorter path for information and gradients.

Instead of:

```text
A → B → C → D → E
```

a skip connection can provide:

```text
A ─────────────────→ E
 \→ B → C → D ─────→ E
```

This can make information and gradients easier to propagate.

Residual/skip connections are widely used in modern neural-network architectures.

---

## 6.4 LSTM

A major solution to the long-term dependency problem is:

> **LSTM — Long Short-Term Memory**

LSTM was specifically designed to make it easier for recurrent networks to preserve useful information for longer periods.

---

# 7. LSTM — Long Short-Term Memory

## Definition

**LSTM (Long Short-Term Memory)** is a special type of recurrent neural network that uses a memory cell and gates to control what information should be:

- Added to memory
- Kept in memory
- Removed from memory
- Used for the current output

In simple words:

> **LSTM gives the recurrent network a more controlled way to decide what information to remember and what information to forget.**

---

# 8. Real Example of LSTM

Consider:

> **"I went to the restaurant yesterday. The food was excellent, but the service was slow."**

The model may need to remember:

```text
food → excellent
service → slow
```

An LSTM can use its gates to control which information remains useful.

For example:

```text
Previous memory
      ↓
   LSTM gates
      ↓
Keep useful information
Forget unnecessary information
Add important new information
      ↓
Updated memory
```

The important point is that LSTM does not simply force every piece of previous information to be remembered.

It **learns what information is important to keep or discard.**

---

# 9. Main Parts of LSTM

LSTM mainly contains:

## 9.1 Cell State

The **cell state** is a pathway through the LSTM that carries information across time steps.

Think of it as:

> **A long-term memory pathway.**

It allows important information to travel through the sequence.

---

## 9.2 Hidden State

The **hidden state** represents information that the LSTM exposes from the current step and passes to the next step.

A simple distinction:

```text
Cell state  → long-term memory pathway
Hidden state → current/output-related information
```

This is a simplified explanation; internally, both interact through the LSTM equations.

---

# 10. LSTM Gates

LSTM uses gates to control information.

The three main gates are:

1. Forget Gate
2. Input Gate
3. Output Gate

---

## 10.1 Forget Gate

### Definition

The **forget gate** decides how much of the previous cell-state information should be kept or forgotten.

Simple idea:

```text
Old memory
    ↓
Forget Gate
    ↓
Keep some information
Forget some information
```

### Real Example

Suppose the model has stored:

> "The user lives in Delhi."

Later, the conversation changes to another person.

The old location may no longer be useful.

The forget gate can learn to reduce the importance of outdated information.

---

# 11. Input Gate

### Definition

The **input gate** controls how much new information should be added to the cell state.

Simple idea:

```text
New information
      ↓
Input Gate
      ↓
Important information
      ↓
Memory
```

### Real Example

Suppose the sequence says:

> "The user's favorite programming language is Python."

The input gate can learn that **Python** is useful information to add to the memory.

---

# 12. Output Gate

### Definition

The **output gate** controls how much information from the internal memory should be exposed as the current hidden state/output.

Simple idea:

```text
Memory
  ↓
Output Gate
  ↓
Current hidden state
```

### Real Example

Suppose the model remembers:

> "The user likes Python."

When the next prediction needs that information, the output gate helps determine how much of the stored information should influence the current output.

---

# 13. LSTM in One Simple Flow

```text
                 Previous Cell State
                        │
                        ▼
                 ┌─────────────┐
Previous Hidden →│    LSTM     │← Current Input
     State       │             │
                 │ Forget Gate │
                 │ Input Gate  │
                 │ Output Gate │
                 └─────────────┘
                    │       │
                    ▼       ▼
             New Cell    New Hidden
               State       State
```

The key idea:

```text
Forget → decide what old information to remove

Input  → decide what new information to store

Output → decide what information to expose
```

---

# 14. Problem B — Unstable Training / Exploding Gradient

## Definition

The **exploding gradient problem** occurs when gradients become extremely large during training.

Instead of:

```text
Gradient → very small → 0
```

we can get:

```text
Gradient → very large → ∞
```

This can cause:

- Very large weight updates
- Unstable training
- Loss becoming extremely large
- `NaN` values
- Model parameters becoming unusable

---

# 15. Simple Real Example of Exploding Gradient

Imagine you are learning to control a car.

Suppose your correction system normally says:

```text
Move steering by +0.1
```

But suddenly it calculates:

```text
Move steering by +100
```

The correction is far too large.

The next correction becomes even worse.

In training, a similar thing can happen:

```text
Normal gradient
      ↓
Large gradient
      ↓
Huge weight update
      ↓
Bad prediction
      ↓
Even larger gradient
      ↓
Unstable training
```

This is the basic idea behind exploding gradients.

---

# 16. Why Does Exploding Gradient Happen?

During BPTT, gradients are repeatedly multiplied through many time steps.

If the effective factors are greater than 1, the product can become extremely large.

For example:

```text
2 × 2 × 2 × 2 × 2
= 32
```

With many more multiplications:

```text
2 × 2 × 2 × ... 
→ extremely large
```

Therefore, the gradient can explode.

---

# 17. Solutions for Exploding Gradient

## 17.1 Gradient Clipping

### Definition

**Gradient clipping** limits the size of gradients before the optimizer uses them to update the model's weights.

Simple idea:

```text
Huge gradient
     ↓
Gradient Clipping
     ↓
Controlled gradient
     ↓
Weight update
```

For example, if a gradient becomes:

```text
1000
```

and the chosen clipping limit is:

```text
10
```

the training process prevents that gradient from causing an excessively large update.

### Important

Gradient clipping does **not make the gradient disappear**.

Its purpose is to:

> **Prevent excessively large gradients from causing unstable weight updates.**

---

# 18. Controlled Learning Rate

The **learning rate** controls how large the model's weight updates are.

A very large learning rate can make training unstable.

Example:

```text
Learning rate = 1.0
```

may produce very large updates in some situations.

A smaller learning rate:

```text
Learning rate = 0.001
```

can make updates more controlled.

But:

> **A smaller learning rate alone does not solve exploding gradients.**

Gradient clipping is specifically designed to control excessively large gradients.

---

# 19. LSTM and Exploding Gradients

LSTM can help with gradient-flow problems because its architecture provides a more controlled path for information and gradients.

However:

> **LSTM is not a magical guarantee that exploding gradients can never happen.**

Gradient clipping and appropriate optimization settings can still be useful when training LSTMs.

---

# 20. Vanishing vs Exploding Gradient

| Problem | What happens? | Main effect | Common solution |
|---|---|---|---|
| Vanishing Gradient | Gradient becomes too small | Difficult to learn long-term dependencies | LSTM/GRU, suitable architecture, initialization |
| Exploding Gradient | Gradient becomes too large | Unstable training | Gradient clipping, suitable learning rate, initialization |

---

# 21. Long-Term Dependency vs Exploding Gradient

These two problems should not be confused.

### Long-Term Dependency

Question:

> **Can the model learn relationships between information that is far apart in the sequence?**

Example:

```text
"The boy who lived in London for ten years ...
... speaks with a British accent."
```

The model may need information from much earlier in the sequence.

---

### Exploding Gradient

Question:

> **Are the training updates becoming excessively large?**

Example:

```text
Gradient = 0.2
```

is manageable.

But:

```text
Gradient = 100000
```

can cause an enormous weight update.

So:

```text
Long-term dependency
        ↓
Difficulty remembering distant information

Exploding gradient
        ↓
Training becomes unstable because updates become too large
```

---

# 22. Complete Picture

A basic RNN looks like:

```text
Input sequence
     ↓
┌─────────┐
│   RNN   │
└─────────┘
     ↓
Hidden states
     ↓
Prediction
```

During training:

```text
Prediction
    ↓
Loss
    ↓
Backpropagation Through Time
    ↓
Gradients
```

The gradients can have two major problems:

```text
                 Gradients
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
     Too small             Too large
          ↓                   ↓
 Vanishing Gradient    Exploding Gradient
          ↓                   ↓
 Long-term dependency  Unstable training
 problem               problem
          ↓                   ↓
 LSTM / GRU             Gradient clipping
 Better architecture    Controlled LR
 Better initialization  Better initialization
```

---

# 23. Important Terms — Simple Definitions

## Sequential Data

Data where the **order of observations matters**.

Example:

```text
Words in a sentence
Temperature every hour
Stock price every day
```

Changing the order can change the meaning.

---

## Context

Information from surrounding or previous parts of a sequence that helps understand the current part.

Example:

```text
"I did not like the movie."
```

The word **"not"** changes the meaning of **"like"**.

---

## Hidden State

Information carried by an RNN from one time step to another.

```text
Previous hidden state
        ↓
Current input
        ↓
      RNN
        ↓
New hidden state
```

---

## Cell State

The internal memory pathway of an LSTM that carries information across time.

It is especially important for preserving useful information over longer sequences.

---

## BPTT

**Backpropagation Through Time** is the process of applying backpropagation to an RNN while considering its sequence of time steps.

Conceptually, the RNN is treated as if it were unfolded:

```text
t1 → t2 → t3 → t4 → t5
```

Then the error is propagated backward:

```text
t5 → t4 → t3 → t2 → t1
```

---

## Gradient

A gradient tells the optimizer how the model's parameters should change to reduce the loss.

Simple idea:

> **Gradient tells the model which direction and how strongly its parameters should change.**

---

## Vanishing Gradient

Gradient becomes extremely small during backpropagation.

Result:

> Earlier time steps receive very little learning signal.

---

## Exploding Gradient

Gradient becomes extremely large during backpropagation.

Result:

> Weight updates can become extremely large and training can become unstable.

---

## Gradient Clipping

A technique that limits excessively large gradients before they update the model's parameters.

---

## Learning Rate

Controls the size of parameter updates made by the optimizer.

```text
Large learning rate
→ larger updates

Small learning rate
→ smaller updates
```

---

## Weight Initialization

The method used to choose the initial values of neural-network weights before training starts.

Good initialization can make optimization more stable.

---

## Activation Function

A function applied inside a neural network that helps determine the output of a neuron and introduces non-linearity.

Examples:

- ReLU
- tanh
- sigmoid

In traditional/basic RNNs, `tanh` is commonly associated with the hidden-state update.

---

## LSTM

**Long Short-Term Memory** is a type of RNN designed to handle information over longer sequences using a memory cell and gates.

---

## Forget Gate

Controls how much previous memory should be forgotten.

---

## Input Gate

Controls how much new information should be added to memory.

---

## Output Gate

Controls how much information from the memory should influence the current hidden state/output.

---

# 24. Final Mental Model

Remember RNN like this:

```text
RNN
│
├── Designed for sequential data
│
├── Uses previous hidden information
│   to process the current input
│
├── Main training problems
│
│   ├── Vanishing Gradient
│   │      ↓
│   │   Gradient becomes too small
│   │      ↓
│   │   Long-term dependency becomes difficult
│   │      ↓
│   │   LSTM / GRU / better architectures
│   │
│   └── Exploding Gradient
│          ↓
│       Gradient becomes too large
│          ↓
│       Training becomes unstable
│          ↓
│       Gradient clipping
│       Controlled learning rate
│
└── LSTM
       ↓
   Memory + Gates
       ↓
   Forget Gate
   Input Gate
   Output Gate
       ↓
   Better control over long-term information
```

# 25. One-Line Summary

> **RNNs are designed for sequential data because previous context can help understand the current input, but basic RNNs can suffer from vanishing gradients that make long-term dependencies difficult to learn and exploding gradients that make training unstable; LSTMs address long-term information flow using controlled memory and gates, while gradient clipping is commonly used to control exploding gradients.**
