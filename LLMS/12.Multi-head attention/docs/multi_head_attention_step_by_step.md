# Multi-Head Attention 


##  What is Multi-Head Attention?

**Meaning:** Multi-Head Attention runs several attention operations in parallel. Each head receives the same input sequence but uses its own learned Query, Key, and Value projections.

A single attention head creates one way of combining information from the tokens. Multiple heads allow the model to calculate several different attention patterns at the same time.

**Real example:**

Consider:

> "The money is in the bank."

When processing the word **bank**, the model may use information from:
- **Head 1:** nearby words such as "the" and "in".
- **Head 2:** the word "money", which may help identify a financial meaning.

These are only intuitive examples. Heads are not manually assigned fixed jobs. Their patterns are learned during training, and we should not assume that a particular head always performs one linguistic task.

**Why it matters:** A sentence contains different kinds of relationships. Multiple heads give the model several learned ways to gather contextual information.

---

##  First Understand Self-Attention

Self-attention allows each token to look at other tokens in the same input sequence and gather useful information from them.

For example:

> "The animal didn't cross the street because it was tired."

To understand **it**, the model can use information from **animal** and the rest of the sentence.

Self-attention turns the initial token representations into context-aware representations.

### Self-attention vs Multi-Head Attention

| Self-attention | Multi-Head Attention |
|---|---|
| One attention operation (one head) | Several attention heads run in parallel |
| Produces one attention pattern per layer operation | Can produce multiple attention patterns |
| Uses one set of Q, K, V projections | Each head has its own Q, K, V projections |
| One contextual output | Head outputs are combined into one output |

Multi-Head Attention is built from self-attention operations. It is not a completely different idea.

---

##  What Are Query, Key, and Value (Q, K, V)?

Think of a library.

- **Query (Q):** What information am I looking for?
- **Key (K):** What information does each item offer? It is used to calculate how relevant that item is to the query.
- **Value (V):** What information should I collect from that item?

In attention:
- Query represents what a token is looking for.
- Key represents what a token can match against.
- Value contains the information that can be passed forward.

The model learns these representations through trainable weight matrices.

### How are Q, K, and V created?

Let the input embedding matrix be `X`.

The model makes three learned linear projections:

```text
Q = X × WQ
K = X × WK
V = X × WV
```

- `X` = input token representations
- `WQ` = learned Query weight matrix
- `WK` = learned Key weight matrix
- `WV` = learned Value weight matrix

The matrices are learned during training using backpropagation and gradient-based optimization.

**Important:** Q, K, and V are not three different input sentences. They are three learned transformations of the input representations.

---

##  Practical Example: "Money in the Bank"

Use the three tokens:

```text
Money     in     bank
```

Assume each token already has an embedding.

For the word **bank**, the model creates a Query. It compares that Query with the Keys of all tokens:

```text
Query(bank) × Key(money)
Query(bank) × Key(in)
Query(bank) × Key(bank)
```

These comparisons produce attention scores.

For illustration only, suppose the scores are:

| Token being compared with | Example score |
|---|---:|
| Money | 2.0 |
| in | 0.5 |
| bank | 1.0 |

The scores are not actual model measurements; they are made-up numbers to explain the calculation.

A higher score means the query has a stronger match with that key before normalization.

---

##  Step 1 — Calculate Attention Scores

The Query of a token is multiplied by the transpose of the Key matrix.

```text
Scores = Q × Kᵀ
```

Why transpose `K`?

Matrix multiplication requires compatible dimensions. If:
- `Q` has shape `(number of tokens, key dimension)`
- `K` has shape `(number of tokens, key dimension)`

then `Kᵀ` has shape `(key dimension, number of tokens)`. The result has shape:

```text
(number of tokens, number of tokens)
```

Each row tells us how strongly one token's Query matches the Keys of all tokens.

For example, the row for **bank** contains its scores against Money, in, and bank.

---

##  Step 2 — Scale the Scores

The raw dot products are divided by the square root of the key dimension:

```text
Scaled scores = (Q × Kᵀ) / √dk
```

`dk` means the dimension of each Query and Key vector in that head.

**Why scale?**

When the vector dimension becomes large, dot products can become large too. Large values can make Softmax very peaked, which can lead to small gradients and make learning harder.

Scaling helps keep the values in a more manageable range.

---

##  Step 3 — Apply Softmax

Softmax converts the scores in each row into attention weights that sum to 1.

For example, after Softmax, an illustrative row might be:

| Token | Attention weight |
|---|---:|
| Money | 0.60 |
| in | 0.10 |
| bank | 0.30 |
| **Total** | **1.00** |

This means the output for the **bank** query will use:
- 60% of the Value for Money
- 10% of the Value for in
- 30% of the Value for bank

These percentages are illustrative, not actual values from a trained model.

**Important:** Attention weights are not necessarily probabilities that a word has a particular meaning. They are normalized weights used to combine Value vectors.

---

##  Step 4 — Multiply Attention Weights by Values

Now the attention weights are multiplied by the Value matrix:

```text
Head output = Softmax((Q × Kᵀ) / √dk) × V
```

This is called **Scaled Dot-Product Attention**.

Using the example weights:

```text
Output for bank =
0.60 × V(money)
+ 0.10 × V(in)
+ 0.30 × V(bank)
```

The result is a weighted combination of the Value vectors.

This output is a **context-aware representation**. It contains information gathered from the other tokens, according to the attention weights.

---

##  What Is an Attention Head?

An attention head is one complete Scaled Dot-Product Attention operation with its own learned Query, Key, and Value projections.

One head performs:

```text
Q → compare with K → Softmax weights → combine V → head output
```

A head does not simply select one word. It normally creates a weighted combination of information from multiple positions.

---

##  Why Do We Need Multiple Heads?

One head calculates one learned attention pattern at a time.

But a sentence can contain several useful relationships:
- relationships between nearby words
- relationships between a pronoun and the word it refers to
- relationships between a verb and its subject
- relationships between words that help identify meaning

Multiple heads can learn different patterns in parallel.

**Analogy:** Imagine several people reading the same sentence. Each person may notice a different relationship. The model combines the information produced by its heads.

Again, this analogy explains the motivation. It does not mean each head is guaranteed to have a human-readable role.

---

##  Two-Head Attention — Practical Architecture

Now use **two heads** for the same input:

```text
Input: "Money in bank"
        |
        +----------------------+
        |                      |
     HEAD 1                 HEAD 2
   Q1, K1, V1             Q2, K2, V2
        |                      |
  Scaled Dot-Product     Scaled Dot-Product
     Attention              Attention
        |                      |
     Output 1               Output 2
        |                      |
        +----------+-----------+
                   |
              Concatenate
                   |
             Output projection
                   |
             Final output
```

Both heads receive the same input sequence, but they use different learned projection weights.

### Head 1

```text
Q1 = X × WQ1
K1 = X × WK1
V1 = X × WV1

Head1 = Softmax((Q1 × K1ᵀ) / √dk) × V1
```

### Head 2

```text
Q2 = X × WQ2
K2 = X × WK2
V2 = X × WV2

Head2 = Softmax((Q2 × K2ᵀ) / √dk) × V2
```

The projections are different, so the heads can learn different attention patterns.

---

##  Step 5 — Concatenation

After attention is calculated, the output of each head is joined side by side.

For two heads:

```text
Combined = Concat(Head1, Head2)
```

Example:

```text
Head 1 output: [0.2, 0.7]
Head 2 output: [0.4, 0.1]

Concatenated output:
[0.2, 0.7, 0.4, 0.1]
```

Concatenation means joining vectors along their feature dimension. It is not addition or averaging.

If each head produces a vector of size 2, two heads produce a concatenated vector of size 4.

---

##  Step 6 — Output Projection (WO)

After concatenation, the result passes through another learned linear transformation:

```text
MultiHeadOutput = Concat(Head1, Head2) × WO
```

`WO` means the output projection weight matrix.

**Why is it needed?**

- It mixes information from the different heads.
- It transforms the concatenated features into the desired model dimension.
- It lets the model learn how to combine the information from all heads.

The final output is the Multi-Head Attention output.

---

##  Complete Multi-Head Attention Formula

The original Transformer paper expresses Multi-Head Attention as:

```text
head_i = Attention(Q × WQ_i, K × WK_i, V × WV_i)

MultiHead(Q, K, V) =
    Concat(head_1, head_2, ..., head_h) × WO
```

Where:

- `h` = number of attention heads
- `WQ_i` = Query projection matrix for head i
- `WK_i` = Key projection matrix for head i
- `WV_i` = Value projection matrix for head i
- `WO` = final output projection matrix

And:

```text
Attention(Q, K, V) =
    Softmax((Q × Kᵀ) / √dk) × V
```

---

##  Understanding Dimensions (Very Important)

Suppose:

- Model dimension (`d_model`) = 4
- Number of heads (`h`) = 2
- Dimension per head (`d_head`) = 2

Then:

```text
d_head = d_model / h
        = 4 / 2
        = 2
```

For an input with 3 tokens:

| Item | Shape |
|---|---|
| Input X | `(3, 4)` |
| Q for each head | `(3, 2)` |
| K for each head | `(3, 2)` |
| V for each head | `(3, 2)` |
| Attention scores per head | `(3, 3)` |
| Output per head | `(3, 2)` |
| Concatenated output | `(3, 4)` |
| Final output after WO | `(3, 4)` |

Why does the attention score matrix have shape `(3, 3)`?

There are 3 query tokens, and each query is compared with 3 key tokens.

**General rule:** In the common equal-head-size design:

```text
d_head = d_model / number_of_heads
```

For example, if `d_model = 512` and there are 8 heads:

```text
d_head = 512 / 8 = 64
```

Each head works with 64-dimensional Query, Key, and Value representations. Concatenating the 8 outputs gives 512 features.

---

##  What Does "Parallel" Mean?

Parallel means the attention heads calculate their results independently from the same input, rather than waiting for one head to finish before starting the next.

Conceptually:

```text
                 Same input X
               /      |      \
            Head 1  Head 2  Head 3
               \      |      /
                 Concatenate
                      |
                Output projection
```

In actual implementations, the Q, K, and V projections for all heads are often computed using large matrix multiplications and then split into head dimensions. This is an efficient implementation of the same idea.

---

##  Self-Attention, Masked Self-Attention, and Cross-Attention

### Self-Attention
Q, K, and V are produced from the same sequence.

Example: The tokens in "Money in bank" attend to tokens in that same sequence.

### Masked (Causal) Self-Attention
A token is prevented from attending to future tokens.

Example:

```text
Money  → can see Money
in     → can see Money, in
bank   → can see Money, in, bank
```

This is useful for autoregressive language models that generate text from left to right.

### Cross-Attention
Queries come from one sequence, while Keys and Values come from another sequence.

Example: In an encoder-decoder Transformer, decoder queries can attend to the encoder's output representations.

**Remember:** Multi-head attention describes the use of multiple heads. Self-attention, causal self-attention, and cross-attention describe where Q, K, and V come from and what positions are allowed to be attended to.

---

##  Where Does Multi-Head Attention Appear in a Transformer?

In the original Transformer architecture:

- The **encoder** uses multi-head self-attention.
- The **decoder** uses masked multi-head self-attention.
- The original decoder also uses encoder-decoder multi-head attention (cross-attention).

Modern Transformer architectures can use different arrangements, but the core attention idea remains important.

---

##  Advantages

1. **Multiple attention patterns:** Different heads can learn different relationships.
2. **Contextual representations:** Tokens can gather information from other relevant tokens.
3. **Parallel computation:** Heads can be computed in parallel.
4. **Flexible information mixing:** The output projection learns how to combine head outputs.

## 20. Limitations

1. **Computational cost:** Attention requires substantial computation, especially for long sequences.
2. **Memory use:** The attention score matrices can become large as sequence length grows.
3. **Not every head is useful:** Some heads may learn overlapping or less useful patterns.
4. **Hard to interpret:** Attention weights alone do not provide a complete explanation of a model's reasoning.

---

##  Common Confusions — Quick Corrections

| Confusion | Correct understanding |
|---|---|
| Q, K, V are the original input | They are learned projections of input representations |
| Query is multiplied by Value | Query is compared with Key; attention weights then combine Values |
| Softmax is applied to Values | Softmax is applied to scaled attention scores |
| Heads are added together | Their outputs are concatenated, then projected |
| Every head has a fixed job | Head behavior is learned; roles are not guaranteed |
| Two heads mean two input sentences | Both heads process the same input sequence |
| Attention weight is the meaning of a word | It is a weight used to combine Value vectors |

---

##  Full Process — One-Minute Revision

```text
Input token embeddings
        |
        v
Create Q, K, V projections for each head
        |
        v
Calculate Q × Kᵀ
        |
        v
Scale by √dk
        |
        v
Apply Softmax
        |
        v
Multiply attention weights by V
        |
        v
Get the output of each head
        |
        v
Concatenate all head outputs
        |
        v
Apply output projection WO
        |
        v
Final Multi-Head Attention output
```

### Easy way to remember

- **Query:** What am I looking for?
- **Key:** How well does this token match the query?
- **Value:** What information can this token contribute?
- **Attention score:** How strong is the Query-Key match?
- **Softmax:** Convert scores into normalized weights.
- **Weighted sum:** Combine the Value vectors.
- **Multiple heads:** Perform several learned attention operations.
- **Concatenation:** Join the head outputs.
- **Output projection:** Mix the heads and produce the final representation.

---


---

## 24. Important Example: "The Man Saw the Astronomer with the Telescope"

This sentence is **ambiguous**. It can have two different meanings:

> "The man saw the astronomer with the telescope."

### Meaning 1 — The man used the telescope

The man is holding or using the telescope to see the astronomer.

```text
THE MAN
   |
   | uses
   v
TELESCOPE
   |
   | to see
   v
ASTRONOMER
```

Here, **"with the telescope"** describes how the man saw the astronomer.

### Meaning 2 — The astronomer had the telescope

The man saw an astronomer who had a telescope.

```text
THE MAN
   |
   | saw
   v
ASTRONOMER
   |
   | has
   v
TELESCOPE
```

Here, **"with the telescope"** describes the astronomer.

**Key point:** The words are exactly the same, but the relationships between the words are different. This is called **structural ambiguity** (more specifically, a possible prepositional-phrase attachment ambiguity).

### What does "perspective" mean here?

In this example, "perspective" means the two possible ways to connect or interpret the phrase **"with the telescope"**:

- Attach it to the action **saw**: the man used a telescope to see.
- Attach it to **astronomer**: the astronomer had a telescope.

It does not mean that the model literally sees the sentence from two human viewpoints.

### How does self-attention help?

Self-attention lets each token gather information from other tokens in the same sentence.

For example, the token **telescope** can attend to:
- **saw** — a possible connection to the instrument used for seeing.
- **astronomer** — a possible connection to the person who has or uses the telescope.
- **man** — another possible participant in the event.

A simplified conceptual picture:

```text
                 "The man saw the astronomer with the telescope"
                                      |
                           Self-attention relationships
                                      |
                 +--------------------+--------------------+
                 |                                         |
          Possible link                              Possible link
          telescope → saw                       telescope → astronomer
                 |                                         |
          instrument used                         astronomer has it
                 |                                         |
          Meaning 1                                Meaning 2
```

This is a conceptual illustration, not a literal attention map. Actual attention weights are learned numerical values, and they do not by themselves prove which interpretation is correct.

### Can one single self-attention head capture both interpretations?

A **single self-attention head** can connect information across the sentence and may learn a useful pattern for this ambiguity. It is not correct to say that one head is completely unable to represent either interpretation.

However, one head produces one attention-weight pattern for each query token in that layer. It has limited ability to express several different relationship patterns independently at the same time.

For example, for the token **telescope**, one head might place more attention on:
- **saw**, highlighting the instrument interpretation; or
- **astronomer**, highlighting the possession/person interpretation.

It can assign weight to both, but those weights are part of one combined attention pattern. The head does not produce two fully separate contextual views just because the sentence is ambiguous.

### How can Multi-Head Attention help?

Multi-Head Attention uses multiple heads in parallel. Each head has its own learned Query, Key, and Value projections.

For teaching purposes, imagine two heads learning different patterns:

```text
              INPUT SENTENCE
"The man saw the astronomer with the telescope"
                       |
              +--------+--------+
              |                 |
           HEAD 1             HEAD 2
              |                 |
       Focuses more on     Focuses more on
          "saw"             "astronomer"
              |                 |
     Possible instrument    Possible person/
        relationship        possession link
              |                 |
              +--------+--------+
                       |
                  CONCATENATE
                       |
                OUTPUT PROJECTION
                       |
            Combined contextual representation
```

This diagram illustrates a **possible** learned behavior, not a guaranteed assignment of jobs. Real heads are not manually programmed as "instrument head" and "possession head". Their patterns emerge during training, and a model may distribute the information differently.

### Does Multi-Head Attention automatically choose the correct meaning?

No. Multi-Head Attention helps the model represent and combine different relationships, but it does not guarantee that the model will resolve every ambiguity correctly.

To choose the intended meaning, a model may need more context, such as:

- "The man looked through the telescope and saw the astronomer."  
  This strongly supports Meaning 1.
- "The astronomer, who was carrying a telescope, was seen by the man."  
  This strongly supports Meaning 2.

The surrounding context, learned language patterns, and later network layers all contribute to interpretation.

### Remember the difference

| Single-head self-attention | Multi-Head Attention |
|---|---|
| One learned attention pattern per query token in that head | Several learned attention patterns in parallel |
| Can connect words across the sentence | Can represent several kinds of connections at once |
| May combine multiple clues in one pattern | Combines outputs from separate head patterns |
| Does not guarantee correct interpretation | Does not guarantee correct interpretation either |

**Final takeaway:** Self-attention helps tokens gather context from other tokens. Multi-Head Attention extends this by allowing multiple learned attention patterns to be computed in parallel. This can help represent different possible relationships in an ambiguous sentence, but it does not mean that every head has a fixed perspective or that multiple heads always solve the ambiguity.


