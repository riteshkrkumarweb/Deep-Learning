# Transformer Decoder Architecture

The Transformer model revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms. While the Encoder processes the input sequence into rich contextual representations, the Decoder generates the target sequence. As detailed in the deep learning architectural analysis by CampusX, understanding the Decoder requires examining how its internal components interact during the training phase. Unlike inference—where generation occurs autoregressively step-by-step—training processes the target sequence in parallel while applying causal masking to preserve temporal ordering.

This technical blueprint breaks down the multi-stage pipeline of the Transformer Decoder, tracing the transformation of input tokens through embeddings, positional encodings, masked attention, cross-attention, feed-forward networks, and final vocabulary projections.

---

## 1. Macro-Architecture: Encoder-Decoder Duality

At the highest level of abstraction, the Transformer architecture operates as a dual-component processing unit comprising an Encoder stack and a Decoder stack. As outlined in the source lecture, both components are constructed by stacking six identical processing blocks ($N = 6$).

```
[ Input Sequence ] --> Encoder Stack (6 Blocks) ----+
                                                    | (Encoder Embeddings: K, V)
                                                    v
[ Target Sequence ] -> Decoder Stack (6 Blocks) ----+--> Linear + Softmax --> [ Output Tokens ]
```

While all blocks within a stack share an identical architectural layout, each block maintains its own independent set of learnable weights and biases.

### Structural Comparison of Stack Blocks

| Feature / Sub-Layer | Encoder Block ($N = 6$) | Decoder Block ($N = 6$) |
| :--- | :--- | :--- |
| **Input Sequence** | Source text (e.g., English sentence "We are friends") | Target text (e.g., Hindi translation "हम दोस्त हैं") |
| **First Sub-Layer** | Standard Multi-Head Self-Attention | Masked Multi-Head Self-Attention (Causal Masking) |
| **Second Sub-Layer** | Position-Wise Feed-Forward Network | Cross-Attention (Encoder-Decoder Attention) |
| **Third Sub-Layer** | N/A | Position-Wise Feed-Forward Network |
| **Residual Connections** | Present across all sub-layers with LayerNorm | Present across all sub-layers with LayerNorm |
| **Operational Mode** | Fully bidirectional parallel processing | Parallel in training (masked); Autoregressive in inference |

---

## 2. Target Sequence Preparation & Embedding Pipeline

Before target tokens can enter the Decoder stack, the raw target text undergoes four sequential input operations designed to convert discrete textual units into dense vector representations containing both semantic and positional information.

```
Target Text ("हम दोस्त हैं")
         |
         v
[ 1. Right-Shift Operation ] ----> Add <START> token ("<START> हम दोस्त हैं")
         |
         v
[ 2. Tokenization Layer ] -------> Break into 4 discrete tokens: [<START>, "हम", "दोस्त", "हैं"]
         |
         v
[ 3. Input Embedding Layer ] ----> Map tokens to 512-dim vectors (e1, e2, e3, e4)
         |
         v
[ 4. Positional Encoding ] ------> Add 512-dim position vectors (p1, p2, p3, p4)
         |
         v
[ Output Vectors ] --------------> x1, x2, x3, x4 (Input to Decoder Block 1)
```

### Step 1: Right-Shift Operation
The target sequence (such as the Hindi translation *"हम दोस्त हैं"*) is right-shifted by prepending a special start token (`<START>`). This shift acts as a trigger flag during training, signaling the start of sentence generation and ensuring that predicting the first real word is conditioned on an initial prompt token. The transformed input sequence becomes:

$$\text{Sequence} = [\text{<START>}, \text{"हम"}, \text{"दोस्त"}, \text{"हैं"}]$$

### Step 2: Tokenization
The shifted target sequence is broken down into discrete processing units or tokens. Depending on the chosen tokenizer, these may represent words, sub-words, or characters. For an example sentence with four tokens, the tokenized output is represented as:

$$T = [T_1, T_2, T_3, T_4] = [\text{<START>}, \text{"हम"}, \text{"दोस्त"}, \text{"हैं"}]$$

### Step 3: Input Token Embedding
Each token is mapped to a continuous dense vector representation using a trainable embedding lookup table. In the standard Transformer configuration established by Vaswani et al. and referenced in the source, the embedding dimension is set to $d_{\text{model}} = 512$. Each token $T_i$ yields an embedding vector $e_i \in \mathbb{R}^{512}$:

$$E = [e_1, e_2, e_3, e_4] \quad \text{where } e_i \in \mathbb{R}^{512}$$

### Step 4: Positional Encoding Integration
Because self-attention mechanisms do not contain inherent recurrence or convolution, they are invariant to token ordering. To preserve sequential structure, positional encodings of dimension $d_{\text{model}} = 512$ are generated for each position $i \in \{1, 2, 3, 4\}$. These vectors ($p_1, p_2, p_3, p_4$) are combined with the embedding vectors via element-wise addition:

$$x_i = e_i + p_i \quad \text{where } x_i \in \mathbb{R}^{512}$$

The resulting input vectors $X = [x_1, x_2, x_3, x_4]$ represent the final enriched inputs fed directly into the first Decoder block.

---

## 3. Internal Mechanics of a Single Decoder Block

Each of the six stacked Decoder blocks processes input vectors through three primary sub-layers, each bounded by residual connections and Layer Normalization.

```
Input Vectors (x1, x2, x3, x4)
         |
         +---------------------------------------+ (Residual Connection 1)
         v                                       |
[ Sub-Layer 1: Masked Self-Attention ]           |
         |                                       |
         v                                       v
[ Add & Layer Normalization ] <------------------+
         | (z'1, z'2, z'3, z'4)
         |
         +---------------------------------------+ (Residual Connection 2)
         v                                       |
[ Sub-Layer 2: Cross-Attention ] <---------------+--- (Encoder Embeddings: K, V)
         |                                       |
         v                                       v
[ Add & Layer Normalization ] <------------------+
         | (zc'1, zc'2, zc'3, zc'4)
         |
         +---------------------------------------+ (Residual Connection 3)
         v                                       |
[ Sub-Layer 3: Feed-Forward Network (FFNN) ]     |
         |                                       |
         v                                       v
[ Add & Layer Normalization ] <------------------+
         |
         v
Output Vectors (y'1, y'2, y'3, y'4) --> Passes to Next Decoder Block
```

### Sub-Layer 1: Masked Multi-Head Self-Attention

The first sub-layer applies self-attention over the target sequence vectors. However, to maintain the causal autoregressive property during training, a look-ahead mask is applied. This prevents any token at position $t$ from attending to future tokens at positions $t' > t$.

> **Causal Masking Constraint Rule**:
> - Token 1 (`<START>`): Attends **only** to `<START>`.
> - Token 2 (`"हम"`): Attends to `<START>` and `"हम"`.
> - Token 3 (`"दोस्त"`): Attends to `<START>`, `"हम"`, and `"दोस्त"`.
> - Token 4 (`"हैं"`): Attends to `<START>`, `"हम"`, `"दोस्त"`, and `"हैं"`.

This operation transforms input vectors $[x_1, x_2, x_3, x_4]$ into masked contextual vectors $[z_1, z_2, z_3, z_4]$.

#### Residual Connection & Layer Normalization (Add & Norm 1)
To prevent gradient vanishing and ensure training stability across deep architectures, a residual skip connection adds the original inputs $x_i$ to the masked attention outputs $z_i$:

$$z^{\text{temp}}_i = x_i + z_i$$

Next, Layer Normalization computes the mean ($\mu$) and variance ($\sigma^2$) across the 512 features for each vector independently, rescaling the features to maintain stable distributions:

$$z'_i = \text{LayerNorm}(x_i + z_i) = \gamma \left( \frac{(x_i + z_i) - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta$$

This yields normalized vectors $[z'_1, z'_2, z'_3, z'_4] \in \mathbb{R}^{512}$.

---

### Sub-Layer 2: Cross-Attention (Encoder-Decoder Attention)

Cross-attention establishes the communication bridge between the source sequence (processed by the Encoder) and the target sequence (being generated by the Decoder). It computes similarity scores between every target token and every source token.

#### Query, Key, and Value Vector Routing

Unlike standard self-attention where Queries ($Q$), Keys ($K$), and Values ($V$) originate from the same sequence, Cross-Attention routes vectors from two distinct sources:

*   **Queries ($Q$)**: Generated by projecting the normalized outputs of the Decoder's masked attention layer ($z'_1, z'_2, z'_3, z'_4$).
*   **Keys ($K$) & Values ($V$)**: Generated by projecting the contextual embeddings produced by the final block of the Encoder stack.

```
Decoder Masked Output (z') ----> Query Linear Projection -----> Q Vectors
Encoder Stack Output ----------> Key & Value Projections ------> K, V Vectors
```

Using $Q$, $K$, and $V$, scaled dot-product attention computes cross-sequence contextual embeddings $[zc_1, zc_2, zc_3, zc_4]$.

#### Residual Connection & Layer Normalization (Add & Norm 2)
A second residual connection combines the masked attention outputs $z'_i$ with the cross-attention outputs $zc_i$, followed by layer normalization:

$$zc'_i = \text{LayerNorm}(z'_i + zc_i) \quad \text{where } zc'_i \in \mathbb{R}^{512}$$

---

### Sub-Layer 3: Position-Wise Feed-Forward Network (FFNN)

The third sub-layer is a fully connected position-wise feed-forward neural network applied independently and identically to each vector position. The network introduces non-linear feature transformations through a two-layer expansion-contraction architecture.

#### Architectural Specifications of the FFNN

1.  **Layer 1 (Expansion)**:
    *   Input Dimension: 512
    *   Hidden Neurons: 2048
    *   Activation Function: ReLU ($\text{ReLU}(x) = \max(0, x)$)
    *   Weight Matrix: $W_1 \in \mathbb{R}^{512 \times 2048}$
    *   Bias Vector: $b_1 \in \mathbb{R}^{2048}$
2.  **Layer 2 (Contraction)**:
    *   Input Dimension: 2048
    *   Output Neurons: 512
    *   Activation Function: Linear (Identity)
    *   Weight Matrix: $W_2 \in \mathbb{R}^{2048 \times 512}$
    *   Bias Vector: $b_2 \in \mathbb{R}^{512}$

#### Matrix Tensor Transformation Tracking

When processing a batch of 4 sequence tokens, the inputs are stacked into a matrix $Z_{\text{cross}} \in \mathbb{R}^{4 \times 512}$. The FFNN processing steps progress as follows:

1.  **First Linear Projection & ReLU**:
    $$H = \text{ReLU}(Z_{\text{cross}} \cdot W_1 + b_1) \quad \text{Shape: } (4 \times 512) \cdot (512 \times 2048) + (2048) \to (4 \times 2048)$$
2.  **Second Linear Projection**:
    $$Y = H \cdot W_2 + b_2 \quad \text{Shape: } (4 \times 2048) \cdot (2048 \times 512) + (512) \to (4 \times 512)$$

This expansion to 2048 dimensions and subsequent contraction back to 512 enables the network to learn rich non-linear feature combinations at each sequence position.

#### Residual Connection & Layer Normalization (Add & Norm 3)
The FFNN outputs $[y_1, y_2, y_3, y_4]$ are combined with the input vectors $zc'_i$ via a residual connection and normalized:

$$y'_i = \text{LayerNorm}(zc'_i + y_i) \quad \text{where } y'_i \in \mathbb{R}^{512}$$

The resulting normalized vectors $[y'_1, y'_2, y'_3, y'_4]$ represent the final output of Decoder Block 1.

---

## 4. Decoder Block Stacking & Sequential Context Accumulation

The output vectors $[y'_1, y'_2, y'_3, y'_4]$ produced by Decoder Block 1 are passed directly as inputs to Decoder Block 2. This process repeats sequentially through all six blocks in the stack:

$$\text{Decoder Block 1} \longrightarrow \text{Decoder Block 2} \longrightarrow \dots \longrightarrow \text{Decoder Block 6}$$

While the architectural operations within each block are identical, every block contains distinct, unshared parameter weights ($W_1^{(l)}, b_1^{(l)}, W_2^{(l)}, b_2^{(l)}, W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}$ for layer index $l \in \{1, \dots, 6\}$).

The final block (Block 6) emits the accumulated contextual representations:

$$Y^f = [y^f_1, y^f_2, y^f_3, y^f_4] \quad \text{where } y^f_i \in \mathbb{R}^{512}$$

---

## 5. Output Projection Head: Vocabulary Projection & Probability Generation

The output representations $Y^f$ emitted by Decoder Block 6 must be translated into word predictions over the target language vocabulary. This is accomplished by the Output Head, which consists of a Linear Projection Layer followed by a Softmax Layer.

```
Decoder Block 6 Output Vectors (Yf) [Shape: 4 x 512]
                        |
                        v
     [ Linear Layer: W3 (512 x V), Bias b3 (V) ]
                        |
                        v
           Unnormalized Logits [Shape: 4 x V]
                        |
                        v
                 [ Softmax Layer ]
                        |
                        v
      Target Token Probabilities [Shape: 4 x V]
                        |
                        v
Prediction Selection (Argmax / Cross-Entropy Loss Calculation)
```

### 1. Linear Projection Layer

The Linear Layer maps each 512-dimensional output vector to a high-dimensional space corresponding to the vocabulary size $V$. The vocabulary size $V$ represents the total count of unique target language tokens present across the training dataset (for example, $V = 10,000$ unique words derived from 5,000 sentence pairs).

*   **Weight Matrix**: $W_3 \in \mathbb{R}^{512 \times V}$
*   **Bias Vector**: $b_3 \in \mathbb{R}^{V}$

For an input matrix $Y^f \in \mathbb{R}^{4 \times 512}$, the matrix multiplication yields unnormalized logit scores $L$:

$$L = Y^f \cdot W_3 + b_3 \quad \text{Shape: } (4 \times 512) \cdot (512 \times V) + (V) \to (4 \times V)$$

Each row in $L$ contains $V$ unnormalized real-valued scores, where each score corresponds to a specific vocabulary token.

### 2. Softmax Normalization Layer

To convert unnormalized logits into normalized probability distributions, the Softmax activation function is applied row-wise across all $V$ vocabulary dimensions. For token position $i$ and vocabulary word index $j \in \{1, \dots, V\}$:

$$P(w_j \mid T_{\le i}) = \text{Softmax}(L_{i, j}) = \frac{\exp(L_{i, j})}{\sum_{k=1}^{V} \exp(L_{i, k})}$$

The output matrix $P \in \mathbb{R}^{4 \times V}$ provides a complete probability distribution over all $V$ words for each token position:

$$\sum_{j=1}^{V} P_{i, j} = 1.0 \quad \text{for each position } i \in \{1, 2, 3, 4\}$$

### Target Token Matching and Training Workflow

During parallel training, each row's probability distribution is compared against the true ground-truth target token using Cross-Entropy Loss:

| Input Token Position | Decoder Input Vector | Expected Target Word | Model Output Goal |
| :--- | :--- | :--- | :--- |
| **Position 1** | $y^f_1$ (derived from `<START>`) | `"हम"` | Predict highest probability for vocabulary index of `"हम"` |
| **Position 2** | $y^f_2$ (derived from `"हम"`) | `"दोस्त"` | Predict highest probability for vocabulary index of `"दोस्त"` |
| **Position 3** | $y^f_3$ (derived from `"दोस्त"`) | `"हैं"` | Predict highest probability for vocabulary index of `"हैं"` |
| **Position 4** | $y^f_4$ (derived from `"हैं"`) | `<END>` | Predict highest probability for end-of-sequence token `<END>` |

---

## 6. Comprehensive Tensor Shape Tracking Matrix

The table below summarizes the exact tensor dimensions, operation types, and data flow across the complete Transformer Decoder prediction pipeline during training for a target sequence of batch length $N_{\text{tokens}} = 4$, model dimension $d_{\text{model}} = 512$, feed-forward hidden dimension $d_{\text{ff}} = 2048$, and vocabulary size $V$.

| Pipeline Stage | Operation / Layer | Input Tensor Shape | Transformation Parameters | Output Tensor Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Target Prep** | Tokenization & Right-Shift | $4 \text{ text tokens}$ | Prepend `<START>` flag | $4 \text{ discrete tokens}$ |
| **Target Prep** | Token Embedding Lookup | $4 \times 1$ | Lookup Table $\mathbb{R}^{V \times 512}$ | $4 \times 512$ |
| **Target Prep** | Positional Encoding Add | $4 \times 512$ | Position Matrix $\mathbb{R}^{4 \times 512}$ | $4 \times 512$ ($X$) |
| **Decoder Block** | Masked Self-Attention | $4 \times 512$ ($X$) | Causal Mask Matrix ($4 \times 4$) | $4 \times 512$ ($Z$) |
| **Decoder Block** | Add & LayerNorm 1 | $X (4 \times 512) + Z (4 \times 512)$ | Gain $\gamma$, Bias $\beta \in \mathbb{R}^{512}$ | $4 \times 512$ ($Z'$) |
| **Decoder Block** | Cross-Attention | $Z' (4 \times 512), \text{Enc} (N_{\text{src}} \times 512)$ | $W_Q, W_K, W_V \in \mathbb{R}^{512 \times 512}$ | $4 \times 512$ ($Z_c$) |
| **Decoder Block** | Add & LayerNorm 2 | $Z' (4 \times 512) + Z_c (4 \times 512)$ | Gain $\gamma$, Bias $\beta \in \mathbb{R}^{512}$ | $4 \times 512$ ($Z'_c$) |
| **Decoder Block** | FFNN Expansion (ReLU) | $4 \times 512$ ($Z'_c$) | $W_1 \in \mathbb{R}^{512 \times 2048}, b_1 \in \mathbb{R}^{2048}$ | $4 \times 2048$ ($H$) |
| **Decoder Block** | FFNN Contraction (Linear) | $4 \times 2048$ ($H$) | $W_2 \in \mathbb{R}^{2048 \times 512}, b_2 \in \mathbb{R}^{512}$ | $4 \times 512$ ($Y$) |
| **Decoder Block** | Add & LayerNorm 3 | $Z'_c (4 \times 512) + Y (4 \times 512)$ | Gain $\gamma$, Bias $\beta \in \mathbb{R}^{512}$ | $4 \times 512$ ($Y'$) |
| **Stack Loops** | Repeat Blocks 2 to 6 | $4 \times 512$ ($Y'$) | 5 Sets of Unshared Block Weights | $4 \times 512$ ($Y^f$) |
| **Output Head** | Linear Vocabulary Projection | $4 \times 512$ ($Y^f$) | $W_3 \in \mathbb{R}^{512 \times V}, b_3 \in \mathbb{R}^{V}$ | $4 \times V$ ($L$) |
| **Output Head** | Softmax Normalization | $4 \times V$ ($L$) | Row-wise Exponential Normalization | $4 \times V$ ($P$) |
