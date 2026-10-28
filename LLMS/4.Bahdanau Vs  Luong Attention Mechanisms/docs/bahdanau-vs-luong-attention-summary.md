# Bahdanau vs. Luong Attention

## 1. Overview

The transcript explains two attention mechanisms used with encoder–decoder models for Neural Machine Translation (NMT):

- **Bahdanau Attention**, also called **Additive Attention**
- **Luong Attention**, also called **Multiplicative Attention** in the transcript

Both mechanisms help the decoder focus on the most relevant encoder hidden states while generating each output word. The transcript introduces them as useful background for understanding the attention ideas that later inspire Transformer self-attention.

> These notes summarize the uploaded transcript. They do not attempt to add material beyond it.

## 2. The Encoder–Decoder Problem

Neural Machine Translation converts a sentence from one language into another. The example in the transcript is translating **“Turn off the lights”** into Hindi.

A basic encoder–decoder architecture works as follows:

1. The **encoder** reads the input sentence one word at a time.
2. It creates hidden states and compresses the sentence into a **context vector** (a numerical representation).
3. The **decoder** uses that representation to generate the translated sentence one word at a time.

### Limitation of the basic approach

The basic architecture relies on a single context vector to represent the whole input sentence. For long sentences or paragraphs, this vector may not preserve enough information. The decoder can then produce lower-quality translations. The transcript describes this as a bottleneck.

## 3. The Core Idea of Attention

Attention reduces the need to rely on just one fixed context vector.

Instead, the encoder makes its hidden states available to the decoder. At each decoding step, the model assigns different weights to those hidden states and combines them into a new **context vector**.

For example, when translating “Turn off the lights”:

- To generate **“lights”**, the decoder should give strong attention to the encoder state associated with “lights”.
- To generate **“off”**, it may give more attention to the states associated with “turn” and “off”.

The model calculates attention weights, written as **α (alpha)**. These weights indicate how much each encoder hidden state contributes to the current output step.

### Context vector

For decoder step \(i\), the context vector is:

\[
c_i = \sum_j \alpha_{ij}h_j
\]

- \(c_i\): context vector for decoder step \(i\)
- \(h_j\): encoder hidden state at input position \(j\)
- \(\alpha_{ij}\): attention weight for encoder position \(j\) at decoder step \(i\)

The attention weights for a decoder step are obtained by applying Softmax to the alignment scores:

\[
\alpha_{ij} =
\frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}
\]

Here, \(e_{ij}\) is the alignment score between the relevant decoder state and encoder hidden state. Softmax normalizes the scores into weights that sum to 1.

## 4. Bahdanau Attention (Additive Attention)

### Main idea

Bahdanau Attention calculates an alignment score using two inputs:

1. The encoder hidden state \(h_j\)
2. The **previous** decoder hidden state \(s_{i-1}\)

The general relationship is:

\[
e_{ij} = f(h_j, s_{i-1})
\]

Rather than choosing a fixed hand-written similarity function, Bahdanau Attention uses a feed-forward neural network to learn the alignment scores.

### Alignment-score equation

The transcript gives this form:

\[
e_{ij} =
v^T \tanh\left(W[s_{i-1};h_j]+b\right)
\]

- \(s_{i-1}\): previous decoder hidden state
- \(h_j\): encoder hidden state at position \(j\)
- \([s_{i-1};h_j]\): concatenation of the two vectors
- \(W\), \(v\), and \(b\): learned parameters
- \(\tanh\): non-linear activation
- \(e_{ij}\): unnormalized alignment score

### How it works

1. The encoder reads the input and produces hidden states such as \(h_1,h_2,h_3,h_4\).
2. At decoder step \(i\), pair the previous decoder state \(s_{i-1}\) with each encoder state \(h_j\).
3. Pass each pair through the feed-forward alignment network to calculate scores \(e_{ij}\).
4. Apply Softmax to the scores to obtain attention weights \(\alpha_{ij}\).
5. Use the weighted sum of encoder states to calculate \(c_i\).
6. In the described architecture, the decoder LSTM uses the context vector along with its previous state and the previous output to produce the next word and update its state.
7. Repeat for the next output step.

### Important architecture detail

The alignment network is shared across decoder time steps. Its parameters are learned during training through backpropagation. The transcript calls this a **time-distributed fully connected neural network**.

### Key characteristics

- Uses the previous decoder hidden state \(s_{i-1}\).
- Uses a feed-forward neural network to calculate alignment scores.
- Is called **Additive Attention** in the transcript.
- Learns which encoder states are relevant at each decoding step.

## 5. Luong Attention (Multiplicative Attention)

### Main idea

Luong Attention uses the **current** decoder hidden state \(s_i\), rather than the previous state \(s_{i-1}\), to calculate alignment scores.

In the version described in the transcript, it uses a dot product instead of a feed-forward neural network:

\[
e_{ij} = s_i^T h_j
\]

If the vectors are \(s_i=[a,b,c,d]\) and \(h_j=[e,f,g,h]\), their dot product is:

\[
e_{ij}=ae+bf+cg+dh
\]

The result is a single scalar alignment score. Softmax converts the scores into attention weights, which are then used to calculate the context vector.

### How it works

1. The encoder produces its hidden states \(h_1,h_2,\ldots,h_n\).
2. The decoder calculates its current hidden state \(s_i\).
3. Calculate the dot product of \(s_i\) with each encoder hidden state \(h_j\).
4. Apply Softmax to the resulting scores to obtain \(\alpha_{ij}\).
5. Calculate the context vector \(c_i\) as the weighted sum of encoder hidden states.
6. Concatenate \(c_i\) with \(s_i\) at the **output stage**.
7. Pass the combined representation through a feed-forward layer and Softmax to predict the next word.
8. Repeat for the next decoding step.

### Key characteristics

- Uses the current decoder hidden state \(s_i\).
- Uses a dot product for alignment scoring in the version described.
- Does not need the Bahdanau-style alignment network for this score calculation.
- Combines the context vector with the decoder state at the output stage, according to the transcript.

## 6. Bahdanau vs. Luong: Main Differences

| Feature | Bahdanau Attention | Luong Attention |
|---|---|---|
| Other name used in the transcript | Additive Attention | Multiplicative Attention |
| Decoder state used for scoring | Previous state, \(s_{i-1}\) | Current state, \(s_i\) |
| Alignment-score method | Feed-forward neural network | Dot product in the described version |
| Score equation | \(e_{ij}=v^T\tanh(W[s_{i-1};h_j]+b)\) | \(e_{ij}=s_i^T h_j\) |
| Context-vector use | Fed into the decoder LSTM in the described architecture | Concatenated with the decoder state at the output stage |
| Computational point emphasized | Additional learned alignment-network parameters | Simpler score calculation; described as faster |

The transcript reports that Luong Attention showed better empirical results in the comparison discussed. This is a report from the video, not a claim that Luong Attention is always better in every task or implementation.

## 7. The Shared Calculation Flow

Both methods follow the same high-level pattern:

1. **Score:** Calculate an alignment score \(e_{ij}\) for each encoder state.
2. **Normalize:** Apply Softmax to get attention weights \(\alpha_{ij}\).
3. **Combine:** Calculate the context vector \(c_i=\sum_j\alpha_{ij}h_j\).
4. **Generate:** Use the context information in the decoder’s output process.

The main difference is how the alignment score is calculated and which decoder hidden state is used.

## 8. Essential Terms

- **Encoder hidden state (\(h_j\))**: A numerical representation produced by the encoder at input position \(j\).
- **Decoder hidden state (\(s_i\))**: The decoder's state at output step \(i\).
- **Alignment score (\(e_{ij}\))**: An unnormalized score indicating the relevance of encoder state \(h_j\) to decoder step \(i\).
- **Attention weight (\(\alpha_{ij}\))**: The normalized weight assigned to encoder state \(h_j\) at decoder step \(i\).
- **Context vector (\(c_i\))**: The weighted sum of encoder hidden states used at decoder step \(i\).
- **Softmax**: Converts a set of scores into normalized weights that sum to 1.
- **Additive Attention**: The Bahdanau approach described, using a learned neural network for alignment scoring.
- **Multiplicative Attention**: The Luong approach described, using a dot product for alignment scoring.

## 9. Final Summary

The basic encoder–decoder model can struggle to represent long input sentences with a single context vector. Attention addresses this by letting the decoder create a context vector at every output step, using weighted encoder hidden states.

- **Bahdanau:** scores each encoder state using a feed-forward network that takes the previous decoder state and the encoder state.
- **Luong:** in the version described, scores each encoder state with a dot product against the current decoder state and combines the context vector at the output stage.

Understanding these methods provides useful background for studying attention in Transformer architectures, as the transcript emphasizes.
