# Bahdanau vs. Luong Attention

- **Bahdanau Attention**, also called **Additive Attention**
- **Luong Attention**, also called **Multiplicative Attention** in the transcript

Both mechanisms help the decoder focus on the most relevant encoder hidden states while generating each output word. The transcript introduces them as useful background for understanding the attention ideas that later inspire Transformer self-attention.

> These notes summarize the uploaded transcript. They do not attempt to add material beyond it.

##  The Encoder–Decoder Problem

Neural Machine Translation converts a sentence from one language into another. The example in the transcript is translating **“Turn off the lights”** into Hindi.

A basic encoder–decoder architecture works as follows:

1. The **encoder** reads the input sentence one word at a time.
2. It creates hidden states and compresses the sentence into a **context vector** (a numerical representation).
3. The **decoder** uses that representation to generate the translated sentence one word at a time.

### Limitation of the basic approach

The basic architecture relies on a single context vector to represent the whole input sentence. For long sentences or paragraphs, this vector may not preserve enough information. The decoder can then produce lower-quality translations. The transcript describes this as a bottleneck.

##  The Core Idea of Attention

Attention reduces the need to rely on just one fixed context vector.

Instead, the encoder makes its hidden states available to the decoder. At each decoding step, the model assigns different weights to those hidden states and combines them into a new **context vector**.

For example, when translating “Turn off the lights”:

- To generate **“lights”**, the decoder should give strong attention to the encoder state associated with “lights”.
- To generate **“off”**, it may give more attention to the states associated with “turn” and “off”.

The model calculates attention weights, written as **attention weights (alpha)**. These weights indicate how much each encoder hidden state contributes to the current output step.

### Context vector

For decoder step The calculation is described in words below., the context vector is:

The calculation is described in words below.

- The calculation is described in words below.: context vector for decoder step The calculation is described in words below.
- The calculation is described in words below.: encoder hidden state at input position The calculation is described in words below.
- The calculation is described in words below.: attention weight for encoder position The calculation is described in words below. at decoder step The calculation is described in words below.

The attention weights for a decoder step are obtained by applying Softmax to the alignment scores:

The calculation is described in words below.

Here, The calculation is described in words below. is the alignment score between the relevant decoder state and encoder hidden state. Softmax normalizes the scores into weights that sum to 1.

##  Bahdanau Attention (Additive Attention)

### Main idea

Bahdanau Attention calculates an alignment score using two inputs:

1. The encoder hidden state The calculation is described in words below.
2. The **previous** decoder hidden state The calculation is described in words below.

The general relationship is:

The calculation is described in words below.

Rather than choosing a fixed hand-written similarity function, Bahdanau Attention uses a feed-forward neural network to learn the alignment scores.

### Alignment-score equation

The transcript gives this form:

The calculation is described in words below.

- The calculation is described in words below.: previous decoder hidden state
- The calculation is described in words below.: encoder hidden state at position The calculation is described in words below.
- The calculation is described in words below.: concatenation of the two vectors
- The calculation is described in words below., The calculation is described in words below., and The calculation is described in words below.: learned parameters
- The calculation is described in words below.: non-linear activation
- The calculation is described in words below.: unnormalized alignment score

### How it works

1. The encoder reads the input and produces hidden states such as The calculation is described in words below..
2. At decoder step The calculation is described in words below., pair the previous decoder state The calculation is described in words below. with each encoder state The calculation is described in words below..
3. Pass each pair through the feed-forward alignment network to calculate scores The calculation is described in words below..
4. Apply Softmax to the scores to obtain attention weights The calculation is described in words below..
5. Use the weighted sum of encoder states to calculate The calculation is described in words below..
6. In the described architecture, the decoder LSTM uses the context vector along with its previous state and the previous output to produce the next word and update its state.
7. Repeat for the next output step.

### Important architecture detail

The alignment network is shared across decoder time steps. Its parameters are learned during training through backpropagation. The transcript calls this a **time-distributed fully connected neural network**.

### Key characteristics

- Uses the previous decoder hidden state The calculation is described in words below..
- Uses a feed-forward neural network to calculate alignment scores.
- Is called **Additive Attention** in the transcript.
- Learns which encoder states are relevant at each decoding step.

##  Luong Attention (Multiplicative Attention)

### Main idea

Luong Attention uses the **current** decoder hidden state The calculation is described in words below., rather than the previous state The calculation is described in words below., to calculate alignment scores.

In the version described in the transcript, it uses a dot product instead of a feed-forward neural network:

The calculation is described in words below.

If the vectors are The calculation is described in words below. and The calculation is described in words below., their dot product is:

The calculation is described in words below.

The result is a single scalar alignment score. Softmax converts the scores into attention weights, which are then used to calculate the context vector.

### How it works

1. The encoder produces its hidden states The calculation is described in words below..
2. The decoder calculates its current hidden state The calculation is described in words below..
3. Calculate the dot product of The calculation is described in words below. with each encoder hidden state The calculation is described in words below..
4. Apply Softmax to the resulting scores to obtain The calculation is described in words below..
5. Calculate the context vector The calculation is described in words below. as the weighted sum of encoder hidden states.
6. Concatenate The calculation is described in words below. with The calculation is described in words below. at the **output stage**.
7. Pass the combined representation through a feed-forward layer and Softmax to predict the next word.
8. Repeat for the next decoding step.

### Key characteristics

- Uses the current decoder hidden state The calculation is described in words below..
- Uses a dot product for alignment scoring in the version described.
- Does not need the Bahdanau-style alignment network for this score calculation.
- Combines the context vector with the decoder state at the output stage, according to the transcript.

##  Bahdanau vs. Luong: Main Differences

| Feature | Bahdanau Attention | Luong Attention |
|---|---|---|
| Other name used in the transcript | Additive Attention | Multiplicative Attention |
| Decoder state used for scoring | Previous decoder state | Current decoder state |
| Alignment-score method | Feed-forward neural network | Dot product in the described version |
| Context-vector use | Fed into the decoder LSTM in the described architecture | Combined with the decoder state at the output stage |
| Computational point emphasized | Uses an additional learned alignment network | Simpler score calculation; described as faster |


The transcript reports that Luong Attention showed better empirical results in the comparison discussed. This is a report from the video, not a claim that Luong Attention is always better in every task or implementation.

##  The Shared Calculation Flow

Both methods follow the same high-level pattern:

1. **Score:** Calculate an alignment score The calculation is described in words below. for each encoder state.
2. **Normalize:** Apply Softmax to get attention weights The calculation is described in words below..
3. **Combine:** Calculate the context vector The calculation is described in words below..
4. **Generate:** Use the context information in the decoder’s output process.

The main difference is how the alignment score is calculated and which decoder hidden state is used.


## Final Summary

The basic encoder–decoder model can struggle to represent long input sentences with a single context vector. Attention addresses this by letting the decoder create a context vector at every output step, using weighted encoder hidden states.

- **Bahdanau:** scores each encoder state using a feed-forward network that takes the previous decoder state and the encoder state.
- **Luong:** in the version described, scores each encoder state with a dot product against the current decoder state and combines the context vector at the output stage.
