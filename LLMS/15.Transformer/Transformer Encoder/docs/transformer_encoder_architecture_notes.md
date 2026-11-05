# Transformer Encoder Architecture

##  Introduction

The Transformer architecture has two main parts:

1.  Encoder
2.  Decoder

This notebook focuses on the **Encoder**, following the explanation in
the provided transcript and the original research paper, *Attention Is
All You Need*.

The encoder processes the input text and produces contextual
representations. These representations are passed to the decoder.

### Main idea

A Transformer encoder is made of multiple identical encoder blocks
stacked one after another.

In the original Transformer architecture:

-   Number of encoder blocks: 6
-   Number of decoder blocks: 6
-   Each encoder block has the same architecture.
-   Each encoder block has its own learned parameters. The weights and
    biases are not shared between blocks.

The number 6 is the configuration used in the original paper, not a
fixed rule for every Transformer.

------------------------------------------------------------------------

##  Inside One Encoder Block

Every encoder block contains two main sub-layers:

1.  Multi-Head Self-Attention
2.  Feed-Forward Neural Network

Each sub-layer is followed by:

-   A residual connection
-   Layer normalization

So, the flow inside one encoder block is:

``` text
Input
  |
  v
Multi-Head Self-Attention
  |
  v
Add Residual Connection
  |
  v
Layer Normalization
  |
  v
Feed-Forward Neural Network
  |
  v
Add Residual Connection
  |
  v
Layer Normalization
  |
  v
Output
```

The output of one encoder block becomes the input to the next encoder
block.

------------------------------------------------------------------------

##  Input Processing

Before the sentence enters the first encoder block, it passes through
three operations.

Example sentence:

**"How are you"**

### Step 1: Tokenization

Tokenization divides a sentence into smaller units called tokens.

For this example, assume word-level tokenization:

``` text
How are you
    |
    v
[How] [are] [you]
```

A tokenizer may also split text into subwords or other token units. The
example here uses whole words for simplicity.

### Step 2: Embedding

A neural network cannot directly process ordinary words as text. The
words must first be represented as numbers.

An embedding layer converts each token into a numerical vector.

In the transcript's example, each token is represented by a vector of
512 numbers.

``` text
How  -> 512-dimensional vector
are  -> 512-dimensional vector
you  -> 512-dimensional vector
```

The three vectors can be arranged into a matrix:

``` text
Shape = 3 x 512
```

Meaning:

-   3 rows: three tokens
-   512 columns: 512 numbers for each token

### Step 3: Positional Encoding

Embedding represents a token, but by itself it does not tell the model
where that token occurs in the sentence.

For example:

``` text
How are you
You are how
```

The order of the words matters.

Positional encoding provides position information. It creates a position
vector for each token position.

In this example, each position vector also has 512 dimensions.

The position vector is added to the corresponding embedding vector:

``` text
Final input vector = Token embedding + Positional encoding
```

The resulting vectors are called position-aware input representations.

``` text
How -> x1 (512 dimensions)
are -> x2 (512 dimensions)
you -> x3 (512 dimensions)
```

The input shape remains:

``` text
3 x 512
```

These vectors enter the first encoder block.

------------------------------------------------------------------------

##  Multi-Head Self-Attention

The first operation inside an encoder block is Multi-Head
Self-Attention.

### Why is self-attention needed?

A normal embedding represents a token, but it does not fully account for
the context in which that token is used.

Consider the word **bank**:

-   River bank: bank means the side of a river.
-   Money bank: bank refers to a financial institution.

The word is the same, but its meaning changes with the surrounding
words.

Self-attention allows a token to use information from other tokens in
the same sentence. This helps create a contextual representation.

### Why Multi-Head Attention?

Multi-Head Attention uses multiple attention heads instead of only one.

Each head can learn different relationships between tokens. Their
results are combined to produce a richer representation.

### Input and output

For the example sentence:

``` text
Input:
x1, x2, x3

Each vector:
512 dimensions
```

After Multi-Head Attention:

``` text
x1 -> z1
x2 -> z2
x3 -> z3
```

Each output vector is still 512-dimensional.

The important difference is that these output vectors now contain
contextual information.

``` text
Input shape  = 3 x 512
Output shape = 3 x 512
```

------------------------------------------------------------------------

##  First Residual Connection and Layer Normalization

After Multi-Head Attention, the encoder uses a residual connection
followed by Layer Normalization.

### Step 1: Residual Connection

The original input is sent through another path, bypassing the attention
operation.

The original input and attention output are added:

``` text
Added output = Original input + Attention output
```

For example:

``` text
x1 + z1 = z1'
x2 + z2 = z2'
x3 + z3 = z3'
```

All vectors have 512 dimensions, so element-wise addition is possible.

### Step 2: Layer Normalization

Layer Normalization is applied separately to each token's vector.

For a 512-dimensional vector:

1.  Calculate the mean of its 512 values.
2.  Calculate the standard deviation.
3.  Normalize the values.
4.  Apply the learnable parameters gamma and beta.

The output vectors are:

``` text
z1_norm
z2_norm
z3_norm
```

Each vector still has 512 dimensions.

### Why Layer Normalization?

According to the transcript, its main purpose here is to help stabilize
the training process by keeping values in a normalized range.

------------------------------------------------------------------------

##  Feed-Forward Neural Network

After the first Add and Norm operation, the vectors enter the
Feed-Forward Neural Network (FFN).

The transcript describes a two-layer neural network.

### Architecture

``` text
Input layer: 512 dimensions
        |
        v
Dense layer: 2048 neurons
Activation: ReLU
        |
        v
Dense layer: 512 neurons
Activation: Linear
        |
        v
Output: 512 dimensions
```

### First layer: 512 to 2048

The input has 512 features.

The first dense layer has 2048 neurons.

Its parameters are:

-   Weight matrix W1: 512 x 2048
-   Bias vector b1: 2048 values
-   Activation: ReLU

The operation can be written as:

``` text
H = ReLU(X x W1 + b1)
```

For three input tokens:

``` text
Input shape  = 3 x 512
Output shape = 3 x 2048
```

The representation expands from 512 to 2048 dimensions.

### Second layer: 2048 to 512

The second dense layer has 512 neurons.

Its parameters are:

-   Weight matrix W2: 2048 x 512
-   Bias vector b2: 512 values
-   Activation: Linear

The operation is:

``` text
Y = H x W2 + b2
```

The shape changes back:

``` text
Input shape  = 3 x 2048
Output shape = 3 x 512
```

### Why expand and then reduce the dimensions?

The first layer expands the representation. ReLU introduces
non-linearity. The second layer projects the representation back to 512
dimensions.

This gives the network more capacity to learn complex patterns while
keeping the output size compatible with the rest of the encoder.

### Important point

The Feed-Forward Network processes each token representation using the
same layer weights within that encoder block. It does not mix
information between different token positions in the way self-attention
does.

------------------------------------------------------------------------

##  Second Residual Connection and Layer Normalization

After the Feed-Forward Network, the encoder uses another residual
connection and Layer Normalization.

The input to the Feed-Forward Network is also sent through a bypass
path.

The two outputs are added:

``` text
Combined output = FFN output + FFN input
```

For the three tokens:

``` text
z1_norm + y1 = y1'
z2_norm + y2 = y2'
z3_norm + y3 = y3'
```

Then Layer Normalization is applied:

``` text
y1_norm
y2_norm
y3_norm
```

Each output vector remains 512-dimensional.

These vectors become the input to the next encoder block.

------------------------------------------------------------------------

##  Complete Information Flow Through One Encoder Block

Here is the complete journey of the example sentence, "How are you".

``` text
Sentence: How are you
        |
        v
Tokenization
        |
        v
[How] [are] [you]
        |
        v
Embedding
        |
        v
3 vectors, each 512-dimensional
        |
        v
Positional Encoding
        |
        v
Add embedding and position vectors
        |
        v
Input to Encoder Block
        |
        v
Multi-Head Self-Attention
        |
        v
Residual Addition
        |
        v
Layer Normalization
        |
        v
Feed-Forward Network
512 -> 2048 -> 512
        |
        v
Residual Addition
        |
        v
Layer Normalization
        |
        v
Output to Next Encoder Block
```

The same process is repeated through all six encoder blocks in the
original Transformer.

The final output is passed to the decoder.

For this three-token example, the shape remains:

``` text
3 x 512
```

The representation changes as the encoder processes it, but its shape
remains the same in this configuration.

------------------------------------------------------------------------

##  Why Are Residual Connections Used?

The transcript is careful to point out that the original Transformer
paper does not give a detailed explanation of why residual connections
were included.

It discusses two commonly suggested reasons.

### Reason 1: Help stabilize training

Very deep neural networks can suffer from the vanishing gradient
problem.

During backpropagation, gradients can become very small as they travel
through many layers. This makes learning difficult.

Residual connections provide an alternative path for information and
gradient flow. They can help gradients travel through deep networks more
effectively.

### Reason 2: Preserve the original information

Imagine that an attention operation produces a representation that is
less useful than the original input.

The residual path allows the original representation to be added back.

This means the model does not have to rely only on the transformed
output. It can retain useful information from the original input.

**Important:** These are explanations discussed in the transcript, not a
definitive explanation supplied by the original Transformer paper.

------------------------------------------------------------------------

##  Why Is the Feed-Forward Network Used?

The transcript presents a common explanation:

-   Self-attention builds contextual representations by allowing tokens
    to interact.
-   The Feed-Forward Network applies additional transformations to each
    token representation.
-   Its ReLU activation introduces non-linearity.
-   This helps the model learn more complex patterns.

The transcript also discusses a research paper that proposes another
perspective:

**Transformer Feed-Forward Layers Are Key-Value Memories**

The paper suggests that Feed-Forward layers can operate like key-value
memories:

-   Keys correlate with textual patterns in training examples.
-   Values induce a distribution over the output vocabulary.

The transcript does not explore this paper in full, so this should be
treated as an additional research perspective rather than a fully
established explanation in the lesson.

------------------------------------------------------------------------

##  Why Are Multiple Encoder Blocks Stacked?

Human language is complex. A single encoder block may not produce
sufficiently powerful representations.

Stacking multiple blocks gives the model more layers in which to process
and transform the representations.

In the original Transformer:

``` text
Input
  |
  v
Encoder Block 1
  |
  v
Encoder Block 2
  |
  v
Encoder Block 3
  |
  v
Encoder Block 4
  |
  v
Encoder Block 5
  |
  v
Encoder Block 6
  |
  v
Final Encoder Output
  |
  v
Decoder
```

### Are all encoder blocks identical?

They have the same architecture, but their learned parameters are
different.

For example:

-   Block 1 has its own attention and FFN weights.
-   Block 2 has different attention and FFN weights.
-   The same applies to the remaining blocks.

During training, each block's parameters are updated through
backpropagation.

### Why exactly six blocks?

According to the transcript, six blocks were selected based on
experimental results in the original configuration.

Six is not a universal requirement. Different Transformer models can use
different numbers of layers depending on their design and application.

------------------------------------------------------------------------

## 12. Important Dimensions to Remember

For the transcript's example:

  Stage                                   Shape
  ---------------------------------- ----------
  Number of tokens                            3
  Token embedding                       3 x 512
  Positional encoding                   3 x 512
  Input to encoder block                3 x 512
  Multi-Head Attention output           3 x 512
  First Layer Normalization output      3 x 512
  FFN first layer                      3 x 2048
  FFN second layer                      3 x 512
  Final Layer Normalization output      3 x 512

### Feed-Forward Network dimensions

  Component                Dimensions
  -------------------- --------------
  Input features                  512
  First dense layer      2048 neurons
  First activation               ReLU
  Second dense layer      512 neurons
  Second activation            Linear
  Output features                 512

------------------------------------------------------------------------

##  Quick Revision

### Transformer Encoder

-   The encoder is one part of the Transformer architecture.
-   The original Transformer uses six encoder blocks.
-   All blocks have the same architecture.
-   Each block has different learned parameters.

### Input processing

-   Tokenization splits text into tokens.
-   Embedding converts tokens into numerical vectors.
-   Positional Encoding adds information about token positions.

### Inside an encoder block

1.  Multi-Head Self-Attention creates contextual representations.
2.  A residual connection adds the original input.
3.  Layer Normalization normalizes each token representation.
4.  A Feed-Forward Network applies transformations and non-linearity.
5.  Another residual connection adds the FFN input.
6.  Another Layer Normalization produces the block output.

### Important dimensions

``` text
512 -> 2048 -> 512
```

The encoder's output dimension remains 512 in the example.

### Final takeaway

**Self-Attention helps tokens understand their context. The Feed-Forward
Network transforms each token representation. Residual connections
provide bypass paths, and Layer Normalization helps stabilize training.
Multiple encoder blocks build increasingly processed representations.**

------------------------------------------------------------------------
