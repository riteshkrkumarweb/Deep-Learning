# How Backpropagation Works in RNN \| Backpropagation Through Time (BPTT)

## 1. What is Backpropagation in an RNN?

**Backpropagation** is the process used to calculate how much each
weight contributed to the prediction error.

The gradients are then used by an optimizer such as **SGD** or **Adam**
to update the weights.

In an RNN, the network processes a sequence one time step at a time.
Because the same RNN weights are reused at every time step, the error is
propagated backward through those time steps.

This is called:

> **Backpropagation Through Time (BPTT)**

------------------------------------------------------------------------

# 2. Simple RNN Example

Suppose the review is:

``` text
"This movie is good"
```

The words can be represented as:

``` text
This  → x₁
movie → x₂
is    → x₃
good  → x₄
```

The RNN processes them sequentially:

``` text
x₁ → RNN → h₁
          ↓
x₂ → RNN → h₂
          ↓
x₃ → RNN → h₃
          ↓
x₄ → RNN → h₄ → Prediction
```

Here:

-   `xₜ` = input at time step `t`
-   `hₜ` = hidden state at time step `t`
-   `h₀` = initial hidden state
-   `h₄` = final hidden state

------------------------------------------------------------------------

# 3. Forward Propagation

During the forward pass, information moves from the first word toward
the last word.

The basic RNN equation is:

``` text
hₜ = f(Wₓₕ xₜ + Wₕₕ hₜ₋₁ + bₕ)
```

Where:

  Symbol   Meaning
  -------- ------------------------------------
  `xₜ`     Current input
  `hₜ₋₁`   Previous hidden state
  `hₜ`     Current hidden state
  `Wₓₕ`    Input-to-hidden weights
  `Wₕₕ`    Hidden-to-hidden/recurrent weights
  `bₕ`     Hidden bias
  `f`      Activation function

For example:

``` text
h₁ = f(Wₓₕx₁ + Wₕₕh₀ + b)

h₂ = f(Wₓₕx₂ + Wₕₕh₁ + b)

h₃ = f(Wₓₕx₃ + Wₕₕh₂ + b)

h₄ = f(Wₓₕx₄ + Wₕₕh₃ + b)
```

The final hidden state can be used to make the prediction:

``` text
ŷ = Softmax(Wₕᵧh₄ + bᵧ)
```

------------------------------------------------------------------------

# 4. Calculate the Loss

The model produces a prediction `ŷ`.

Suppose:

``` text
Actual label:     Positive
Prediction:       Negative
```

The prediction is wrong, so we calculate a **loss**.

For classification, a common loss is cross-entropy:

``` text
L = -Σ yᵢ log(ŷᵢ)
```

The loss tells us how wrong the prediction was.

``` text
Prediction
    ↓
Calculate Loss
    ↓
How wrong was the model?
```

------------------------------------------------------------------------

# 5. What is BPTT?

BPTT means:

> **Backpropagation Through Time**

The RNN is conceptually **unrolled** across its time steps.

Instead of looking at one RNN cell:

``` text
x → RNN → output
```

we look at the repeated computation:

``` text
x₁ → RNN → h₁ → RNN → h₂ → RNN → h₃ → RNN → h₄ → Prediction
```

The same RNN weights are reused at every time step.

During the backward pass, the gradients move in the opposite direction:

``` text
Loss
 ↓
h₄
 ↓
h₃
 ↓
h₂
 ↓
h₁
```

Therefore, the error travels backward **through time steps**.

------------------------------------------------------------------------

# 6. Why is it called "Through Time"?

Each position in the sequence is considered a **time step**.

For our review:

``` text
Time step 1 → This
Time step 2 → movie
Time step 3 → is
Time step 4 → good
```

Forward:

``` text
t₁ → t₂ → t₃ → t₄ → Prediction
```

Backward:

``` text
t₄ → t₃ → t₂ → t₁
```

The gradients move backward through these time steps.

That is why it is called:

> **Backpropagation Through Time**

------------------------------------------------------------------------

# 7. Forward Pass vs Backward Pass

## Forward Pass

Information moves forward:

``` text
x₁ → h₁ → h₂ → h₃ → h₄ → ŷ → Loss
```

## Backward Pass

Gradients move backward:

``` text
Loss → h₄ → h₃ → h₂ → h₁
```

Simple idea:

``` text
FORWARD:

Input → Hidden States → Prediction → Loss


BACKWARD:

Loss → Gradients → Earlier Hidden States
```

------------------------------------------------------------------------

# 8. Why Does the Error Reach Earlier Words?

Consider:

``` text
"This movie is not good"
```

The word `not` can affect the meaning of `good`.

The RNN needs to learn this relationship.

During the forward pass:

``` text
This → movie → is → not → good → Prediction
```

During BPTT:

``` text
Prediction/Loss
      ↓
    good
      ↓
     not
      ↓
     is
      ↓
   movie
      ↓
    This
```

The gradient can therefore carry information about the final error
toward earlier time steps.

------------------------------------------------------------------------

# 9. The Chain Rule

Backpropagation uses the **chain rule of calculus**.

For example, the gradient reaching an earlier hidden state can involve:

``` text
∂L/∂h₁
```

and can be expressed through a chain of derivatives:

``` text
∂L/∂h₁
=
∂L/∂h₄
×
∂h₄/∂h₃
×
∂h₃/∂h₂
×
∂h₂/∂h₁
```

The important idea is:

> The gradient is passed backward through multiple time steps using the
> chain rule.

------------------------------------------------------------------------

# 10. The Important Shared Weights

An RNN does **not** normally create completely different weights for
every word.

The same weights are reused.

For example:

``` text
x₁ → [RNN]
       ↑
      Wₕₕ

x₂ → [RNN]
       ↑
      Wₕₕ

x₃ → [RNN]
       ↑
      Wₕₕ

x₄ → [RNN]
       ↑
      Wₕₕ
```

The same `Wₕₕ` is used at each time step.

The same is true for the other RNN parameters such as `Wₓₕ` and the
biases.

------------------------------------------------------------------------

# 11. Gradients for Shared Weights

Because the same weights are used repeatedly, each time step contributes
to the gradient.

Conceptually:

``` text
Gradient from t₁
       +
Gradient from t₂
       +
Gradient from t₃
       +
Gradient from t₄
       ↓
Total gradient
```

For a shared parameter:

``` text
∂L/∂W = Σₜ ∂Lₜ/∂W
```

The accumulated gradient is then used to update the shared weight.

------------------------------------------------------------------------

# 12. Updating the Weights

After calculating the gradients, an optimizer updates the weights.

The basic gradient descent equation is:

``` text
W_new = W_old - η(∂L/∂W)
```

Where:

-   `W_new` = updated weight
-   `W_old` = old weight
-   `η` = learning rate
-   `∂L/∂W` = gradient of the loss with respect to the weight

The goal is to change the weights so that the loss becomes smaller.

------------------------------------------------------------------------

# 13. Complete BPTT Process

The complete process is:

``` text
1. Input sequence
       ↓
2. Forward propagation through time
       ↓
3. Generate prediction
       ↓
4. Calculate loss
       ↓
5. Backpropagate the error
       ↓
6. Move gradients backward through time steps
       ↓
7. Calculate gradients for shared weights
       ↓
8. Accumulate the gradients
       ↓
9. Optimizer updates the weights
       ↓
10. Repeat during training
```

------------------------------------------------------------------------

# 14. Complete Architecture

``` text
                    FORWARD PASS
                         →

x₁ ──→ [RNN] ──→ h₁ ──→ [RNN] ──→ h₂ ──→ [RNN] ──→ h₃ ──→ [RNN] ──→ h₄
        ↑                  ↑                  ↑                  ↑
        │                  │                  │                  │
        └──────────────────┴──────────────────┴──────────────────┘
                       SAME RNN WEIGHTS

                                                                  ↓
                                                             Prediction
                                                                  ↓
                                                                Loss
                                                                  │
                    ← ← ← ← BACKWARD PASS ← ← ← ← ← ← ← ← ← ← ┘

                  Gradients flow from later time steps
                  toward earlier time steps.
```

------------------------------------------------------------------------

# 15. Vanishing Gradient Problem

During BPTT, gradients can involve repeated multiplication.

If the values are smaller than `1`, the gradient can become smaller and
smaller.

Example:

``` text
0.5 × 0.5 × 0.5 × 0.5

= 0.0625
```

With many time steps:

``` text
Gradient
   ↓
smaller
   ↓
smaller
   ↓
smaller
   ↓
almost 0
```

This is called the:

> **Vanishing Gradient Problem**

When the gradient becomes extremely small, the RNN has difficulty
learning relationships between distant time steps.

------------------------------------------------------------------------

# 16. Exploding Gradient Problem

The opposite can also happen.

If repeated multiplication makes the gradient larger:

``` text
2 × 2 × 2 × 2

= 16
```

With many time steps, it can become extremely large.

This is called:

> **Exploding Gradient Problem**

Gradient clipping is commonly used to help control very large gradients.

------------------------------------------------------------------------

# 17. Why LSTM and GRU Are Useful

Traditional RNNs can have difficulty learning long-term dependencies
because of vanishing and exploding gradients.

Architectures such as:

-   **LSTM (Long Short-Term Memory)**
-   **GRU (Gated Recurrent Unit)**

were designed to handle long-term dependencies more effectively.

They use gating mechanisms to control the flow of information.

------------------------------------------------------------------------

# 18. BPTT in One Simple Diagram

``` text
FORWARD:

"This" → "movie" → "is" → "good"
   ↓         ↓        ↓       ↓
  h₁        h₂       h₃      h₄
                              ↓
                         Prediction
                              ↓
                            Loss


BACKWARD:

Loss
 ↓
Gradient
 ↓
h₄
 ↓
h₃
 ↓
h₂
 ↓
h₁
 ↓
Update shared weights
```

------------------------------------------------------------------------

# 19. The Most Important Things to Remember

◇ **RNN processes a sequence one time step at a time.**

◇ **The hidden state carries information from previous time steps.**

◇ **The same RNN weights are reused across time steps.**

◇ **The forward pass moves information from earlier to later time
steps.**

◇ **The loss measures how wrong the prediction is.**

◇ **BPTT sends gradients backward through the time steps.**

◇ **The chain rule is used to calculate the gradients.**

◇ **Gradients from different time steps contribute to the shared
weights.**

◇ **The optimizer uses these gradients to update the weights.**

◇ **Repeated gradient multiplication can cause vanishing or exploding
gradients.**

------------------------------------------------------------------------

# 20. One-Line Definition

> **Backpropagation Through Time (BPTT) is the method of training an RNN
> by propagating the prediction error backward through its sequence of
> time steps to calculate gradients and update the shared weights.**

------------------------------------------------------------------------

# 21. Easy Memory Trick

``` text
FORWARD
Input → Hidden → Hidden → Hidden → Prediction → Loss

BACKWARD
Loss → Gradient → Hidden → Hidden → Hidden

UPDATE
Gradients → Weights → Better Prediction
```

**Remember:**

> **Forward = information moves through time.**\
> **Backward = gradients move back through time.**\
> **Update = weights are changed to reduce the loss.**
