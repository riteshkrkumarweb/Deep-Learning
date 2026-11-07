# Transformer Decoder: Autoregressive Generation and Masked Self-Attention

## 1. Big Picture

A GPT-style Transformer decoder behaves differently during training and
inference:

- **During training:** the correct target sequence is already available.
  The decoder can calculate predictions for all positions in parallel.
- **During inference:** future tokens are unknown. The decoder generates
  one token at a time, using the tokens already available.

The key mechanism that makes parallel training possible without allowing
future-token information is **masked self-attention**, also called
**causal self-attention**.

------------------------------------------------------------------------

## 2. What Is Inference Time?

**Meaning:** Inference is the stage when a trained model is used to make
predictions or generate output for new input.

**Example:**

A trained language model receives:

`The weather today is`

It predicts a next token, then uses the expanded sequence to predict
another token.

Inference is when we use the trained model. Training is when the model
learns its parameters from data.

------------------------------------------------------------------------

## 3. What Does Autoregressive Mean?

**Meaning:** An autoregressive model generates a sequence one step at a
time. Each new token is predicted using the tokens that are already
available, including previously generated tokens.

**Example:**

Prompt: `The cat`

1.  Input: `The cat` → predict `sat`
2.  Input: `The cat sat` → predict `on`
3.  Input: `The cat sat on` → predict `the`
4.  Input: `The cat sat on the` → predict `mat`

The model generates tokens sequentially because it needs the earlier
generated tokens as context for later predictions.

**Why it matters:** At inference time, the model does not know future
tokens. It must generate a token before it can use that token to help
predict the next one.

------------------------------------------------------------------------

## 4. What Is Teacher Forcing?

**Meaning:** Teacher forcing is a training technique in which the
decoder receives the actual correct previous tokens from the training
data, rather than its own previous predictions.

**Example:**

Suppose the correct target sentence is:

`I am fine`

Imagine the model predicts `you` when the correct first token is `I`.

With teacher forcing, the next decoder input still uses the correct
token from the dataset, `I`, not the model’s incorrect prediction,
`you`.

The model’s predictions are compared with the correct target tokens
using a loss function. The loss is then used to update the model’s
weights through backpropagation.

**Important:** Teacher forcing does not give the model the answer to the
position it is currently predicting. It supplies the correct previous
tokens; the model must still predict the next token.

------------------------------------------------------------------------

## 5. Why Is Sequential Training Slow?

If training copied inference exactly, the decoder would have to:

1.  Predict the first token.
2.  Wait for that prediction.
3.  Feed it back into the decoder to predict the next token.
4.  Repeat for every token.

For long sequences and large datasets, this is slow because the
decoder’s computation would have to run step by step.

During training, however, the correct target sequence is already
available. With teacher forcing, we can provide the shifted target
sequence as input and calculate predictions for all positions in
parallel.

But parallel processing creates a problem: ordinary self-attention could
allow an earlier position to use information from future target tokens.
That would be **future-token information leakage**.

Masked self-attention solves this problem.

------------------------------------------------------------------------

## 6. Self-Attention: A Quick Review

Self-attention lets each token build a contextual representation by
using information from other tokens in the same sequence.

For example, the meaning of `bank` depends on its context:

- `river bank` — the side of a river
- `money bank` — a financial institution

Self-attention uses three vectors:

- **Query (Q):** what information a token is looking for.
- **Key (K):** what information a token offers for matching.
- **Value (V):** the information that is combined to create the output.

Basic steps:

1.  The input embeddings are projected into Q, K, and V.
2.  Queries and keys are compared to calculate attention scores.
3.  The scores are scaled by the square root of the key dimension.
4.  Softmax converts the scores into attention weights.
5.  The weights are used to combine the value vectors and create
    contextual representations.

The scaled dot-product attention formula is:

``` text
Attention(Q, K, V) = softmax((Q × Kᵀ) / √d_k) × V
```

In ordinary, unmasked self-attention, a token can attend to tokens both
before and after it.

That is useful for bidirectional representations, but it is not suitable
for a causal language-model decoder.

------------------------------------------------------------------------

## 7. What Is Masked Self-Attention?

**Meaning:** Masked self-attention is self-attention with a causal mask.
The mask prevents a token from attending to future positions.

A token can attend to:

- Previous tokens
- Its own position

It cannot attend to future tokens.

### Causal mask example

For the sequence:

`The cat sat on`

| Query token | The     | cat     | sat     | on      |
|-------------|---------|---------|---------|---------|
| The         | Allowed | Blocked | Blocked | Blocked |
| cat         | Allowed | Allowed | Blocked | Blocked |
| sat         | Allowed | Allowed | Allowed | Blocked |
| on          | Allowed | Allowed | Allowed | Allowed |

Rows represent the token doing the attending. Columns represent the
token being attended to.

The allowed positions form a lower-triangular pattern. Each position can
use itself and earlier positions, but not later ones.

------------------------------------------------------------------------

## 8. How Does the Causal Mask Work?

The causal mask is applied to the attention scores **before Softmax**.

The masked attention calculation is:

``` text
Masked Attention(Q, K, V)
= softmax((Q × Kᵀ) / √d_k + M) × V
```

Here, `M` is the causal mask:

- Allowed positions receive `0`.
- Blocked future positions receive negative infinity (`-∞`).

**Why use negative infinity?**

Softmax turns a score of negative infinity into a probability of zero:

``` text
exp(-∞) = 0
```

Therefore, blocked future positions receive zero attention weight. They
cannot contribute to the token’s output representation.

### Small example

For the first token, `The`, the model must not use `cat`, `sat`, or `on`
to build its representation.

The mask blocks those future positions.

For the second token, `cat`, the model can use `The` and `cat`, but not
`sat` or `on`.

------------------------------------------------------------------------

## 9. Why Do We Need Masked Self-Attention If We Already Have Self-Attention?

This is the central question.

**Self-attention:** Allows tokens to use information from other tokens
in the sequence.

**Masked self-attention:** Adds a restriction: a token can use only
itself and earlier tokens. Future tokens are blocked.

Without the mask, during parallel training, the representation at an
earlier position could use later target tokens. The model would receive
information that it would not have when generating text. This is
future-token information leakage.

With the causal mask:

- All target positions can be processed in parallel during training.
- Each position can use only information available up to that position.
- Future-token leakage is prevented.
- The model learns the causal prediction behaviour it must follow during
  generation.

So masked self-attention gives the decoder the benefits of parallel
training while preserving the correct causal behaviour.

------------------------------------------------------------------------

## 10. Training vs. Inference

| Feature                  | Training                                                    | Inference                                     |
|--------------------------|-------------------------------------------------------------|-----------------------------------------------|
| Goal                     | Learn model parameters                                      | Generate predictions                          |
| Previous tokens supplied | Correct tokens from the training sequence (teacher forcing) | Prompt tokens and previously generated tokens |
| Decoder computation      | Target positions can be processed in parallel               | New tokens are generated sequentially         |
| Causal mask              | Used                                                        | Used                                          |
| Future target tokens     | Present in the training sequence, but blocked by the mask   | Not yet generated                             |

**Important distinction:** The decoder uses causal masking in both
training and inference. The difference is that training can process all
target positions together, while ordinary autoregressive inference must
generate new tokens one at a time.

------------------------------------------------------------------------

## 11. Key Terms

| Term                  | Simple meaning                                                                |
|-----------------------|-------------------------------------------------------------------------------|
| Inference             | Using a trained model to make predictions                                     |
| Autoregressive        | Generating each new token using earlier tokens                                |
| Teacher forcing       | Supplying the correct previous tokens during training                         |
| Self-attention        | Letting tokens use information from other tokens                              |
| Masked self-attention | Self-attention that blocks future positions                                   |
| Causal mask           | The mask that prevents looking ahead                                          |
| Future-token leakage  | Using future target information that would not be available during generation |
| Parallel training     | Processing multiple sequence positions together                               |

------------------------------------------------------------------------

## 12. Questions I Asked

### Q1. What is inference time?

Inference time is when a trained model is used to make predictions or
generate output for new input.

### Q2. What does autoregressive mean?

Autoregressive means generating a sequence one token at a time, where
each new token depends on the previously available tokens.

### Q3. What is teacher forcing?

Teacher forcing is a training technique in which the decoder receives
the correct previous tokens from the training sequence instead of its
own previous predictions.

### Q4. What is masked self-attention?

Masked self-attention is self-attention that blocks each token from
attending to future tokens. It can attend to previous tokens and itself.

### Q5. Why do we need masked self-attention when we already have self-attention?

Ordinary self-attention can use future tokens. During parallel training,
this would let a position access information it would not have during
generation. A causal mask blocks future positions, preventing
information leakage while allowing parallel computation across
positions.

------------------------------------------------------------------------

## 13. Final Revision

Remember these four ideas:

1.  **Inference:** The trained model is being used to generate
    predictions.
2.  **Autoregressive generation:** Generate one token, then use it to
    help generate the next.
3.  **Teacher forcing:** During training, provide the correct previous
    tokens so the decoder does not have to wait for its own predictions.
4.  **Masked self-attention:** Block future tokens so parallel training
    does not leak future information.

### One-line summary

**Teacher forcing makes parallel training possible; causal masking
prevents future-token leakage; autoregressive generation produces tokens
sequentially at inference time.**
