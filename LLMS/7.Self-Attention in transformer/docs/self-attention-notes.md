# Self-Attention in Transformers: Notes

## 1. Why Self-Attention Matters

- Generative AI is spreading rapidly, and the **Transformer** architecture sits at its core.
- At the center of the Transformer is **Self-Attention**.
- The video builds the concept from first principles.

---

## 2. Recap: Why Word Embeddings Fall Short

In NLP, words must be converted to numbers: One-Hot Encoding, Bag of Words, Word Embeddings.

Word embeddings (Word2Vec, GloVe) capture semantic meaning, but they are **static**.

**The problem with static embeddings:**

| Sentence | Meaning of "Bank" |
|---|---|
| "Money Bank Grows" | Financial institution |
| "River Bank Flows" | Side of a river |

A static embedding gives "Bank" the **same vector** in both sentences, even though the meanings differ.

---

## 3. The Need for Dynamic Contextual Embeddings

- We need embeddings that **change based on the surrounding words**.
- **Self-Attention** converts static embeddings into dynamic contextual embeddings.

```
"Humans Love Smartphones"

Static (e1, e2, e3)  ->  [ Self-Attention Block ]  ->  Contextual (y1, y2, y3)
```

---

## 4. Building Self-Attention from First Principles

Represent each word as a weighted mix of the words in its context:

- Sentence 1: `Bank = 0.2 * Money + 0.7 * Bank + 0.1 * Grows`
- Sentence 2: `Bank = 0.4 * River + 0.5 * Bank + 0.1 * Flows`

Replacing words with their embedding vectors:

```
New Embedding(Money) = 0.7 * E(Money) + 0.2 * E(Bank) + 0.1 * E(Grows)
```

The coefficients (0.7, 0.2, 0.1) represent **similarity** between word embeddings, computed with **dot products**.

### Normalization via Softmax

Raw dot products (s21, s22, s23) can be positive or negative and of any size. Softmax converts them into weights that sum to 1:

```
w21 = exp(s21) / [exp(s21) + exp(s22) + exp(s23)]

y_bank = w21 * E(Money) + w22 * E(Bank) + w23 * E(Grows)
```

---

## 5. Advantages and Limitations of the Simple Approach

**Advantages**
- **Parallelization:** all words are processed together using matrix operations, which suits GPUs.

**Limitations**
- **Sequential information loss:** word order is ignored (addressed later by Positional Encoding).
- **No learnable parameters:** it produces general contextual embeddings, not task-specific ones (e.g., for Machine Translation or Sentiment Analysis).

---

## 6. Query, Key, and Value

Each word embedding plays **three roles**:

| Role | Purpose |
|---|---|
| **Query (Q)** | Asks other words, "How similar are you to me?" |
| **Key (K)** | The identity that gets queried |
| **Value (V)** | The actual content vector used in the weighted sum |

**Analogy: a matrimonial / dating website**

| Website | Self-Attention |
|---|---|
| Autobiography | Original embedding vector |
| Search criteria | Query (Q) |
| Public profile | Key (K) |
| Actual conversation / meeting | Value (V) |

---

## 7. Learnable Linear Transformations

Instead of using the raw embedding for all three roles, project the embedding **E** through three learnable weight matrices:

```
Q = E * W_q
K = E * W_k
V = E * W_v
```

- `W_q`, `W_k`, `W_v` start with **random weights**.
- They are learned during training via **backpropagation**, driven by the task loss.

---

## 7.1 Q&A: Why Do We Need Query, Key, and Value?

**Q1. What goes wrong with the simple approach that has no Q, K, V?**

The same raw embedding is used for everything: to ask "who is similar to me?", to be matched against, and to supply the content. There are **no learnable parameters**, so the output is a *general* contextual embedding. It cannot adapt to a specific task such as Machine Translation or Sentiment Analysis.

**Q2. So why introduce Query, Key, and Value?**

Because one word plays **three different roles** during attention, and a single vector cannot serve all of them well:

- **Query:** the word asks, "How similar are you to me?"
- **Key:** the word offers its identity so others can query it.
- **Value:** the word supplies the actual content that gets mixed into the output.

Giving each role its own vector lets the model treat them separately.

**Q3. How does the dating-website analogy explain this?**

On a matrimonial site, one person has:

- an **autobiography** (the full original profile, i.e. the embedding),
- **search criteria** for what they are looking for (Query),
- a **public profile** that others see and match against (Key),
- the **actual conversation or meeting** once matched (Value).

You search with your criteria, matching happens against public profiles, and what you actually get from a match is the real interaction. Each part has a different purpose, just like Q, K, V.

**Q4. How do Q, K, V become task-specific?**

Each is made by multiplying the embedding with its own learnable matrix: `Q = E * W_q`, `K = E * W_k`, `V = E * W_v`. These matrices start random and are trained by **backpropagation** using the task loss, so the model learns what to look for, what to advertise, and what content to pass along for the task at hand.

**Q5. What is the main benefit in one line?**

Q, K, V add **learnable parameters** to self-attention, turning a fixed, general similarity calculation into a trainable, task-specific one.

---

## 8. Final Matrix Formulation

1. Input embeddings matrix **X** with shape `N x d`
2. Compute `Q = X * W_q`, `K = X * W_k`, `V = X * W_v`
3. Similarity scores: `Scores = Q * K^T`
4. Attention weights: `Weights = Softmax(Scores)`
5. Output: 

```
Y = Softmax(Q * K^T) * V
```

The whole pipeline runs **in parallel on GPUs** and forms the core of modern Transformer architectures.

---

## Quick Summary

- Static embeddings give one vector per word, regardless of context.
- Self-attention makes each word's embedding a **context-weighted** combination of all words in the sentence.
- Weights come from **dot-product similarity + softmax**.
- Adding **Q, K, V with learnable matrices** makes the process task-specific and trainable.
- Final formula: **Y = Softmax(Q·Kᵀ) · V**


Your question is:

If the dot product already calculates a similarity score between two vectors, why do we need Query, Key, Value and their weight matrices?

The answer is that the dot product only calculates a score. It does not decide by itself what information to search for, what information to match, or what information to collect.

Let's understand this step by step.
1. First, imagine a sentence
Consider:

"Money grows in the bank."

The Transformer needs to understand the relationships between these words.
For example:
- Does money relate to bank?
- Does money relate to grows?
- Which words should contribute information to the representation of money?
This is where Q, K and V come in.
2. Why do we need Query, Key and Value?
Think of a library.
Imagine you want to find information about Machine Learning.
Transformer	Library example
Query (Q)	Your search question
Key (K)	Labels or descriptions of books
Value (V)	Actual information inside the books

How does it work?
1. Query: What information am I looking for?
2. Key: Which information matches my query?
3. Value: What information should I retrieve?
The dot product compares the Query with each Key to calculate matching scores.

But the actual information is collected from the Values.