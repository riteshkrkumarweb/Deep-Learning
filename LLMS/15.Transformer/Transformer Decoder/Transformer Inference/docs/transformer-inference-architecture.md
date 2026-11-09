# Complete Technical Guide to Transformer Inference Architecture

Understanding Transformer inference requires examining how a trained neural sequence-to-sequence model transitions from offline training to live prediction. While training operates in a parallelized, teacher-forced regime, inference executes dynamically and autoregressively. This document presents a comprehensive, step-by-step technical analysis of the Transformer prediction pipeline during machine translation.

---

## 1. Experimental Setup & Comparative Paradigm: Training vs. Inference

To analyze inference mechanics concretely, consider a machine translation task converting an English input sentence into a Hindi target sentence. The dataset consists of aligned translation pairs, pre-processed and trained using standard cross-entropy loss over a target vocabulary of size $V$.

* **Input Query (English)**: *"We are friends"*
* **Target Output (Hindi)**: *"हम दोस्त हैं"*

During evaluation, model parameters (weights and biases) are static and locked. The query sentence is provided to the network to generate the Hindi output token-by-token.

### Functional Divergence Between Training and Inference

The primary operational difference between the training phase and the inference phase lies within the Decoder architecture.

| Component | Training Phase Behavior | Inference Phase Behavior | Operational Characteristics |
| :--- | :--- | :--- | :--- |
| **Encoder stack** | Parallel execution across all input tokens | Parallel execution across all input tokens | Identical execution flow; generates static contextual embeddings |
| **Decoder stack** | Non-autoregressive (Parallel execution) | Autoregressive (Sequential step-by-step execution) | Target sequence is unavailable at inference; tokens generated iteratively |
| **Decoder Masking** | Causal Masking applied to prevent looking ahead | Causal Masking applied to preserve training distribution alignment | Prevents attention leakage across time steps |
| **Input Feedback** | Target tokens fed simultaneously via teacher forcing | Previously predicted tokens fed back iteratively | Sequence length expands by $+1$ token at each time step |

The Encoder processes the full input sequence once. In contrast, the Decoder runs iteratively in a loop, consuming generated tokens step-by-step until an End of Sentence token is emitted.

---

## 2. The Encoder Pipeline: Parallel Representation Generation

The Encoder architecture operates identically during inference as it does during training. Its role is to convert the raw source text into rich contextual vector representations.

```
Input Tokens ("We", "are", "friends")
   │
   ▼
Word Embedding Layer (512-dim) + Positional Encoding
   │
   ▼
6 × Encoder Blocks (Multi-Head Self-Attention + Add & LayerNorm + Feed-Forward)
   │
   ▼
Contextual Vectors (3 vectors of 512-dim) ──► Fanned to Decoder Cross-Attention
```

### Step-by-Step Processing Breakdown

1. **Tokenization**: The input string *"We are friends"* is split into individual tokens: $T_{\text{enc}} = [\text{"We"}, \text{"are"}, \text{"friends"}]$.
2. **Dense Embedding**: Each token is mapped through a learned lookup matrix into a $512$-dimensional vector.
3. **Positional Encoding Addition**: Element-wise addition of $512$-dimensional positional vectors injects order information into the sequence, yielding initial representation vectors.
4. **Stacked Encoder Block Iterations**:
   * The vectors pass through $6$ identical Encoder blocks.
   * Each block applies Multi-Head Self-Attention to compute contextual relationships among all tokens simultaneously.
   * Outputs pass through Residual Add and Layer Normalization operations, followed by a Feed-Forward Neural Network ($512 \to 2048 \to 512$ dimensions) with ReLU activations.
5. **Contextual Vector Output**: The Encoder outputs $3$ final $512$-dimensional contextual vectors—one for *"We"*, one for *"are"*, and one for *"friends"*. Each vector encapsulates full bidirectional context from the entire input sentence. These vectors remain fixed throughout the decoding loop.

---

## 3. The Autoregressive Decoder Engine: Step 1 (`<SOS>` Initiation)

Since the target output is unknown at inference initialization, decoding must be kicked off using a special Start of Sentence token: $\text{<SOS>}$.

### 3.1 Input Embedding and Positional Encoding
* Token $\text{<SOS>}$ enters the Decoder embedding layer, generating a $512$-dimensional embedding vector.
* A positional encoding vector for position index $0$ is added element-wise.
* This results in input vector $x_1 \in \mathbb{R}^{512}$.

### 3.2 Masked Self-Attention Layer
Vector $x_1$ is projected into Query ($q$), Key ($k$), and Value ($v$) vectors via weight matrices $W_q, W_k, W_v \in \mathbb{R}^{512 \times 512}$:

$$q_1 = x_1 W_q, \quad k_1 = x_1 W_k, \quad v_1 = x_1 W_v$$

Because only a single token ($\text{<SOS>}$) exists at Step 1, self-attention computes the scaled dot-product of $q_1$ and $k_1$:

$$\text{Attention Score} = \text{Softmax}\left(\frac{q_1 \cdot k_1^T}{\sqrt{d_k}}\right) v_1$$

This produces intermediate representation $z_1 \in \mathbb{R}^{512}$. A residual connection adds $x_1$ to $z_1$, followed by Layer Normalization:

$$z_{1\text{norm}} = \text{LayerNorm}(x_1 + z_1)$$

### 3.3 Cross-Attention Mechanics
The Cross-Attention layer bridges the Decoder and Encoder representations:
* **Query Vector**: Derived from the Decoder's normalized self-attention output: $q_{\text{SOS}} = z_{1\text{norm}} W_q^{\text{cross}}$.
* **Key and Value Vectors**: Derived from the Encoder's $3$ contextual vectors ($v_{\text{We}}, v_{\text{are}}, v_{\text{friends}}$):
  * Key vectors: $k_{\text{We}}, k_{\text{are}}, k_{\text{friends}} \in \mathbb{R}^{512}$
  * Value vectors: $v_{\text{We}}, v_{\text{are}}, v_{\text{friends}} \in \mathbb{R}^{512}$

Cross-attention computes similarity scale factors ($w_1, w_2, w_3$) between $q_{\text{SOS}}$ and each Encoder key vector:

$$w_i = \text{Softmax}\left( \frac{q_{\text{SOS}} \cdot k_i^T}{\sqrt{d_k}} \right)$$

The weighted sum over the Encoder Value vectors yields the cross-attention output $z_{c1} \in \mathbb{R}^{512}$:

$$z_{c1} = w_1 v_{\text{We}} + w_2 v_{\text{are}} + w_3 v_{\text{friends}}$$

Following a residual add and Layer Normalization, the representation becomes $z_{c1\text{norm}} = \text{LayerNorm}(z_{1\text{norm}} + z_{c1})$.

### 3.4 Feed-Forward Neural Network (FFNN) and Block Stacking
Vector $z_{c1\text{norm}}$ is passed to a two-layer Feed-Forward network:
1. **Hidden Layer**: $2048$ neurons with ReLU activation: $h_1 = \text{ReLU}(z_{c1\text{norm}} W_1 + b_1)$, where $W_1 \in \mathbb{R}^{512 \times 2048}$.
2. **Output Layer**: $512$ neurons with linear activation: $y_1 = h_1 W_2 + b_2$, where $W_2 \in \mathbb{R}^{2048 \times 512}$.

After residual addition and Layer Normalization, vector $y_{1\text{norm}}$ is passed as input to the next Decoder block. This exact transformation sequence repeats across all $6$ stacked Decoder blocks, yielding final output vector $y_{f1\text{norm}} \in \mathbb{R}^{512}$.

### 3.5 Linear Vocabulary Projection and Token Selection
To convert $y_{f1\text{norm}}$ into a word prediction, it is projected onto the vocabulary dimension $V$:

1. **Linear Projection**: Multiplied by weight matrix $W_{\text{vocab}} \in \mathbb{R}^{512 \times V}$ to produce raw unnormalized logits $L \in \mathbb{R}^V$. Each logit corresponds to one unique Hindi word in the vocabulary.
2. **Softmax Transformation**: Normalizes logits into a probability distribution:

$$P(w_i) = \frac{e^{L_i}}{\sum_{j=1}^{V} e^{L_j}}$$

3. **Greedy Token Selection**: The word with the highest probability is selected. In this example, the model assigns the highest probability to *"हम"* (Hum).

Step 1 concludes with the emission of output token *"हम"*.

---

## 4. Multi-Token Decoding & Causal Masking in Inference (Step 2 Onwards)

In Step 2, the sequence fed into the Decoder expands to include both the initial $\text{<SOS>}$ token and the newly predicted token *"हम"*: $T_{\text{dec}} = [\text{<SOS>}, \text{"हम"}]$.

### 4.1 Input Vector Construction
The Decoder computes embeddings and positional encodings for both tokens, producing two input vectors:
* $x_1 \in \mathbb{R}^{512}$ for $\text{<SOS>}$ at index $0$.
* $x_2 \in \mathbb{R}^{512}$ for *"हम"* at index $1$.

### 4.2 Causal Masking Execution During Inference
A common point of confusion is whether causal masking is necessary during inference, given that past tokens are already generated. **Causal masking is strictly mandatory during inference.**

```
Attention Matrix (Step 2):
                Key: <SOS>    Key: "हम"
Query: <SOS>  [   Score    |   MASKED (-∞) ]
Query: "हम"   [   Score    |     Score     ]
```

1. **Masking Mechanism**: When computing self-attention for $x_1$ and $x_2$, a $2 \times 2$ score matrix is formed. Causal masking forces the upper triangular entry (row $1$, column $2$) to $-\infty$ prior to Softmax calculation.
2. **Information Isolation**:
   * Token $\text{<SOS>}$ attends **only** to itself ($x_1$). Its attention weight toward future token *"हम"* is zeroed out.
   * Token *"हम"* ($x_2$) attends to both $\text{<SOS>}$ ($x_1$) and itself ($x_2$).
3. **Rationale for Masking at Inference**: During training, the Decoder was trained under masked attention constraints. Removing masking during inference would create a severe data distribution shift between training and evaluation, degrading generation quality.

### 4.3 Multi-Token Vector Propagation
The dual vectors $[x_1, x_2]$ propagate through all $6$ Decoder layers:
* Masked Self-Attention outputs $z_1, z_2$.
* Cross-Attention processes $z_1, z_2$ against the $3$ Encoder contextual vectors.
* Feed-Forward networks process both vectors independently.

At the output of the $6\text{th}$ Decoder block, two $512$-dimensional vectors emerge: $[y_{f1\text{norm}}, y_{f2\text{norm}}]$.

### 4.4 The Selective Projection Rule
At Step 2, two vectors arrive at the Linear Projection layer. However, **only the final vector ($y_{f2\text{norm}}$) is projected to the vocabulary distribution**:

* Vector $y_{f1\text{norm}}$ (corresponding to $\text{<SOS>}$) is discarded at this stage because its prediction (*"हम"*) was already executed in Step 1.
* Vector $y_{f2\text{norm}}$ (corresponding to *"हम"*) is multiplied by $W_{\text{vocab}} \in \mathbb{R}^{512 \times V}$ and softmaxxed.
* The token with the highest output probability is selected: *"दोस्त"* (Dost).

---

## 5. Sequential Termination & Pipeline Summary

Decoding proceeds iteratively, appending each new output token to the input sequence of the next step.

### Complete Step-by-Step Translation Walkthrough

1. **Step 1**: Input $[\text{<SOS>}] \implies$ Predicts *"हम"*.
2. **Step 2**: Input $[\text{<SOS>}, \text{"हम"}] \implies$ Predicts *"दोस्त"*.
3. **Step 3**: Input $[\text{<SOS>}, \text{"हम"}, \text{"दोस्त"}] \implies$ Predicts *"हैं"*.
4. **Step 4**: Input $[\text{<SOS>}, \text{"हम"}, \text{"दोस्त"}, \text{"हैं"}] \implies$ Predicts $\text{<EOS>}$ (End of Sentence).

Upon predicting $\text{<EOS>}$, the generation loop terminates immediately. The complete translated output is assembled: **"हम दोस्त हैं"**.

### Tensor Dimensionality and Layer Transformation Summary

| Processing Stage | Input Tensor Shape | Transformation / Operation | Output Tensor Shape | Key Architectural Note |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder Embedding** | $[B, N_{\text{src}}]$ | Lookup + Positional Encoding | $[B, N_{\text{src}}, 512]$ | Parallel processing across full source sentence |
| **Encoder Output** | $[B, N_{\text{src}}, 512]$ | $6 \times$ (Self-Attn + FFNN) | $[B, N_{\text{src}}, 512]$ | Fixed contextual representation for Cross-Attention |
| **Decoder Input (Step $t$)** | $[B, t]$ | Lookup + Positional Encoding | $[B, t, 512]$ | Sequence length $t$ grows by $+1$ each step |
| **Masked Self-Attention** | $[B, t, 512]$ | Query/Key/Value Projection + Causal Mask | $[B, t, 512]$ | Upper triangle masked to $-\infty$ |
| **Cross-Attention** | $[B, t, 512]$ Decoder, $[B, N_{\text{src}}, 512]$ Encoder | Decoder Query $\times$ Encoder Key/Value | $[B, t, 512]$ | Connects target step representation to source context |
| **Decoder FFNN** | $[B, t, 512]$ | $512 \to 2048 \to 512$ linear layers | $[B, t, 512]$ | Applies non-linear transformations via ReLU |
| **Vocabulary Projection** | $[B, 1, 512]$ (Slice of position $t$) | Linear Layer ($W_{\text{vocab}} \in \mathbb{R}^{512 \times V}$) | $[B, 1, V]$ | Only the last position slice is projected |
| **Token Selection** | $[B, 1, V]$ | Softmax + Argmax / Sampling | $[B, 1]$ | Single discrete token emitted per time step |

This step-by-step pipeline highlights the precise balance of parallel representation extraction in the Encoder and sequential autoregressive generation in the Decoder that defines Transformer inference.
