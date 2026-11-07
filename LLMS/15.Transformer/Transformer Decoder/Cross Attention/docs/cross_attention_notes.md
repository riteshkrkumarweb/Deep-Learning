# Cross-Attention in Transformers

## 1. What Is Cross-Attention?

Cross-attention is an attention mechanism that allows a model to connect
two different sequences.

It is especially useful in sequence-to-sequence tasks, where a model
receives one sequence and generates another.

**Simple definition:** Cross-attention helps the output sequence focus
on the relevant parts of the input sequence while generating each output
token.

### Example: English-to-Hindi Translation

Input sentence (English):

> I like eating ice cream.

Output sentence (Hindi):

> मुझे आइसक्रीम खाना पसंद है।

The encoder processes the English sentence. The decoder generates the
Hindi sentence one token at a time.

When the decoder generates a Hindi word, it needs two kinds of
information:

1.  What has the decoder generated so far?
2.  What information is present in the original English sentence?

These two requirements are handled by different attention mechanisms.

------------------------------------------------------------------------

## 2. Why Do We Need Cross-Attention If We Already Have Self-Attention?

Self-attention and cross-attention solve different problems.

### Self-attention in the decoder

The decoder uses masked self-attention to examine the output tokens
generated so far.

For example, if the decoder has generated:

> मुझे आइसक्रीम

Self-attention helps the decoder understand the relationship between
these generated tokens and use that context to generate the next token.

However, decoder self-attention only attends to the decoder's own
sequence. It does not directly retrieve information from the encoder's
output.

### Cross-attention in the decoder

The decoder also needs information from the original input sentence.

Cross-attention creates this connection. It allows the decoder to look
at the encoder's output and focus on relevant input information.

For example:

-   Hindi "आइसक्रीम" can attend strongly to English "ice cream".
-   Hindi "पसंद" can attend strongly to English "like".
-   Hindi "खाना" can attend strongly to English "eating".

**Key idea:** Self-attention provides context from the decoder's own
sequence. Cross-attention provides access to information from the
encoder's sequence.

------------------------------------------------------------------------

## 3. How Does Cross-Attention Work?

Cross-attention is conceptually similar to self-attention. The important
difference is where Query, Key, and Value come from.

  Component   Source in cross-attention
  ----------- ---------------------------
  Query (Q)   Decoder / output sequence
  Key (K)     Encoder / input sequence
  Value (V)   Encoder / input sequence

Remember:

-   **Query:** What information am I looking for?
-   **Key:** What information is available to match against?
-   **Value:** What information can be retrieved?

### Step-by-step process

1.  The encoder processes the input sequence and produces contextual
    representations.
2.  The decoder has its current representations from the previous
    decoder sublayer.
3.  The decoder representations are projected to create Queries.
4.  The encoder outputs are projected to create Keys and Values.
5.  Each Query is compared with the Keys using dot products.
6.  The scores are scaled and passed through softmax to produce
    attention weights.
7.  The attention weights are used to calculate a weighted sum of the
    Values.
8.  The resulting vectors are contextual representations for the decoder
    positions.

The decoder can therefore focus on different parts of the input sequence
at different output positions.

### Cross-attention formula

Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V

For cross-attention:

-   Q is produced from the decoder.
-   K and V are produced from the encoder.

The formula is the same basic scaled dot-product attention formula. The
source of Q differs from the source of K and V.

------------------------------------------------------------------------

## 4. Self-Attention vs Cross-Attention

  -----------------------------------------------------------------------
  Feature                 Self-attention          Cross-attention
  ----------------------- ----------------------- -----------------------
  Sequences involved      One sequence            Two sequences

  Query                   From that sequence      From decoder/output
                                                  sequence

  Key                     From that sequence      From encoder/input
                                                  sequence

  Value                   From that sequence      From encoder/input
                                                  sequence

  Main purpose            Model relationships     Connect output
                          within a sequence       positions to input
                                                  information

  Typical decoder use     Understand previously   Retrieve relevant
                          generated tokens        encoder information
  -----------------------------------------------------------------------

### An easy way to remember

-   **Self-attention:** A sequence looks at itself.
-   **Cross-attention:** One sequence looks at another sequence.

------------------------------------------------------------------------

## 5. What Is Different in the Input?

### Self-attention

Self-attention receives one sequence.

Example:

> We are friends.

Each token has an embedding or a contextual representation from a
previous layer. These representations are used to create Q, K, and V.

### Cross-attention

Cross-attention connects two sequences.

Example:

Input sequence:

> We are friends.

Output sequence:

> हम दोस्त हैं।

The two sequences provide different inputs:

-   The output/decoder sequence provides the Query representations.
-   The input/encoder sequence provides the Key and Value
    representations.

This is the main input-level difference.

------------------------------------------------------------------------

## 6. What Happens to the Output?

Self-attention produces a contextual representation for every position
in its sequence.

Cross-attention produces a contextual representation for every position
in the decoder/output sequence.

Therefore, the number of cross-attention output vectors matches the
number of positions in the decoder sequence entering that
cross-attention layer.

Each output vector is formed by taking a weighted sum of the encoder's
Value vectors.

The attention weights indicate how strongly a decoder position attends
to each encoder position.

**Important:** These weights are learned dynamically from the Query-Key
comparisons. They are not fixed word-to-word translation rules.

------------------------------------------------------------------------

## 7. Connection to Earlier Attention Mechanisms: Bahdanau and Luong Attention

The transcript connects Transformer cross-attention to attention
mechanisms used in earlier RNN-based encoder-decoder models.

In those models:

1.  The encoder processes the input sequence.
2.  The decoder generates output tokens step by step.
3.  At each decoding step, the model compares the current decoder state
    with encoder states.
4.  It calculates attention scores and uses them to form a context
    vector.
5.  The context vector helps the decoder generate the next output token.

The transcript mentions two approaches:

-   **Bahdanau attention:** Uses a small neural network to calculate
    alignment/attention scores.
-   **Luong attention:** Uses a dot-product-based approach for attention
    scoring.

Transformer cross-attention performs a closely related role: decoder
positions attend over encoder representations.

The original Transformer paper refers to this as **encoder-decoder
attention**.

------------------------------------------------------------------------

## 8. Cross-Attention in the Original Transformer Decoder

The original Transformer decoder contains two different attention
sublayers:

1.  **Masked multi-head self-attention:** The decoder attends to earlier
    output positions, without looking at future target tokens.
2.  **Encoder-decoder attention (cross-attention):** The decoder attends
    to the encoder's output.

In cross-attention, the Queries come from the previous decoder sublayer,
while the Keys and Values come from the encoder output.

This lets every decoder position attend over the positions in the input
sequence.

------------------------------------------------------------------------

## 9. Where Is Cross-Attention Used?

Cross-attention is useful when a model needs to connect two different
sequences or modalities.

Examples include:

-   **Machine translation:** English input to Hindi output.
-   **Question answering:** A question attends to information in a
    passage.
-   **Image captioning:** Text-generation positions attend to image
    features.
-   **Text-to-image generation:** Image-generation representations
    attend to text information.
-   **Text-to-speech:** Speech-generation representations attend to text
    representations.

These examples involve connecting information from one source to
another.

------------------------------------------------------------------------

## 10. Is Cross-Attention Like a Prediction Function in Machine Learning?

They are related to the overall prediction process, but they are not the
same thing.

### Prediction function in machine learning

A simple supervised learning model can be represented as:

ŷ = f(X)

-   X = input features
-   f = learned model/function
-   ŷ = predicted output

For example, a house-price model receives house size, location, and
number of bedrooms, then predicts a price.

### Cross-attention in a Transformer

Cross-attention does not directly produce the final predicted token.
Instead, it retrieves and combines relevant information from the encoder
output for each decoder position.

The decoder then processes this information through the remaining
layers. The final output layer produces scores or probabilities over the
vocabulary, which are used to choose the next token.

**Conclusion:** Cross-attention contributes to the prediction process,
but it is not itself the complete prediction function.

------------------------------------------------------------------------

## 11. Questions and Answers

### Q1. What is cross-attention?

Cross-attention is an attention mechanism that lets one sequence focus
on relevant information from another sequence. In an encoder-decoder
Transformer, the decoder uses it to attend to the encoder's output.

### Q2. Why do we need cross-attention if we already have self-attention?

Decoder self-attention only attends to the decoder's own sequence. It
helps the decoder use previously generated tokens. Cross-attention is
needed to give the decoder access to the encoder's input information.

### Q3. Is cross-attention like the prediction function in machine learning?

Not exactly. Cross-attention retrieves relevant information and creates
contextual representations. The final prediction is produced later by
the decoder's output layers.

### Q4. Where do Q, K, and V come from in cross-attention?

Q comes from the decoder. K and V come from the encoder.

### Q5. What is the difference between self-attention and cross-attention?

Self-attention models relationships within one sequence. Cross-attention
models relationships between two sequences.

### Q6. How many output vectors does cross-attention produce?

It produces one output vector for each decoder position entering that
cross-attention layer. Thus, the output sequence length determines the
number of output vectors.

------------------------------------------------------------------------

## 12. Quick Revision

-   Cross-attention connects two different sequences.
-   It is especially useful in sequence-to-sequence tasks.
-   In encoder-decoder Transformers, Q comes from the decoder.
-   K and V come from the encoder.
-   Self-attention looks within a sequence.
-   Cross-attention lets the decoder retrieve information from the
    encoder.
-   Cross-attention produces contextual representations for decoder
    positions.
-   The original Transformer paper calls it encoder-decoder attention.
-   Cross-attention contributes to prediction, but it does not itself
    produce the final token prediction.

**One-line definition to remember:**

Cross-attention allows the decoder to focus on relevant information from
the encoder's output while generating the output sequence.
