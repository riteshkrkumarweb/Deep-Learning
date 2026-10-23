# Bidirectional RNN, BiLSTM, and BiGRU 

##  What is a Bidirectional RNN?

 **Bidirectional Recurrent Neural Network (BiRNN)** processes a sequence in two directions:

1. **Forward RNN:** reads the sequence from left to right.
2. **Backward RNN:** reads the sequence from right to left.

At each time step, the model combines the hidden state from each direction—usually by concatenating them—to produce an output.

### Why use it?

A normal, unidirectional RNN processes inputs in one direction. Its state at a particular time step mainly represents the current input and information from earlier inputs. However, some tasks need information from both the words before and after a particular word.

**Real example:** Consider these sentences:

- “I love Amazon; it’s a great website.”
- “I love Amazon; it’s a beautiful river.”

The word **Amazon** can refer to an organization or a geographical location. The later words—*website* or *river*—help determine the intended meaning. A model that can use context from both sides can make a more informed prediction.

##  Quick recap: Unidirectional RNN

A unidirectional RNN reads the sequence one step at a time, for example:

`x₁ → x₂ → x₃ → x₄`

Information moves forward through the sequence. An output at a later time step can use information carried from earlier time steps, but the standard forward RNN does not have access to later inputs when it processes an earlier time step.

### Limitation

If the correct label for a word depends on words that come after it, a forward-only model may lack important context at that point.

##  How a Bidirectional RNN works

A BiRNN uses two separate recurrent networks.

### Forward direction

The forward network reads from left to right:

`x₁ → x₂ → x₃ → x₄`

It produces forward hidden states:

`h₁_forward, h₂_forward, h₃_forward, h₄_forward`

Each state summarizes information from the current input and the preceding sequence.

### Backward direction

The backward network reads from right to left:

`x₄ → x₃ → x₂ → x₁`

It produces backward hidden states aligned with the corresponding sequence positions:

`h₁_backward, h₂_backward, h₃_backward, h₄_backward`

Each backward state carries information from the current position and the sequence to its right.

### Combine the hidden states

At each position, concatenate the two hidden states:

`hₜ = [hₜ_forward ; hₜ_backward]`

The combined representation can then be passed to an output layer to predict a label for that position.

**Additional explanation:** Concatenation means joining vectors end to end. If each direction produces a hidden state with 5 values, concatenating the two states produces a vector with 10 values.

##  Keras implementation

Keras provides a `Bidirectional` wrapper. It can wrap a `SimpleRNN`, `LSTM`, or `GRU` layer.

```python
from keras.layers import Bidirectional, SimpleRNN, LSTM, GRU

# Bidirectional SimpleRNN
model.add(Bidirectional(SimpleRNN(5), input_shape=(...)))

# Bidirectional LSTM (BiLSTM)
model.add(Bidirectional(LSTM(5)))

# Bidirectional GRU (BiGRU)
model.add(Bidirectional(GRU(5)))
```

### What does each line mean?

- `from keras.layers import ...`: imports the recurrent layers and the bidirectional wrapper.
- `Bidirectional(...)`: runs a recurrent layer in both forward and backward directions.
- `SimpleRNN(5)`: creates a recurrent layer with 5 units per direction.
- `LSTM(5)`: creates an LSTM layer with 5 units per direction.
- `GRU(5)`: creates a GRU layer with 5 units per direction.
- `model.add(...)`: adds the configured layer to a Keras Sequential model.
- `input_shape=(...)`: placeholder notation in the transcript, not a complete runnable shape. Replace `...` with the actual input dimensions.

### Important distinction: `return_sequences`

For sequence-labeling tasks, you often need an output at every time step. In Keras, you generally use `return_sequences=True` for a recurrent layer when the next layer needs the full sequence of outputs.

Example:

```python
model.add(Bidirectional(LSTM(5, return_sequences=True)))
model.add(Bidirectional(GRU(4)))
```

In this example, the first bidirectional LSTM returns an output for every time step, so the next recurrent layer can process the whole sequence. The final bidirectional GRU, with the default `return_sequences=False`, returns the final combined representation rather than an output at every time step.

**Note:** This is an additional example to explain Keras behavior; the transcript's original snippets do not specify `return_sequences`.

##  Why do the number of parameters increase?

A bidirectional layer has separate trainable parameters for its forward and backward recurrent networks. Therefore, it generally has more parameters than the corresponding one-directional layer.

The transcript gives an example in which a simple RNN has 190 weights and biases, while its bidirectional version has 380. This doubling applies to that example; the exact parameter count depends on the input dimension, number of units, and layer type.

**Practical impact:**
- More parameters to train.
- Potentially greater training time and memory use.
- A greater risk of overfitting when the dataset is small, so regularization or dropout may be useful.

##  Applications

The transcript lists these application areas:

1. **Named Entity Recognition (NER):** identify names of people, organizations, places, and other entities in text.
2. **Part-of-Speech (POS) tagging:** assign grammatical labels such as noun, verb, or adjective to words.
3. **Machine translation:** use sequence context when converting text from one language to another.
4. **Sentiment analysis:** classify text sentiment using contextual information.
5. **Time-series forecasting:** the transcript mentions examples such as stock-price prediction and weather forecasting.

**Caution about forecasting:** A bidirectional model is appropriate only when the inputs it uses are available at prediction time. Using future observations that would not be available in a real forecasting situation can cause data leakage. This is an additional practical clarification.

##  Limitations and drawbacks

###  Increased model complexity

Because the model contains forward and backward recurrent networks, it uses more parameters than a comparable one-directional network. This can increase training cost and overfitting risk.

**Possible response:** Use suitable regularization, dropout, and validation to assess generalization.

###  Latency in real-time applications

The transcript notes that a bidirectional model can create a delay in real-time speech recognition because it may need the full sequence before it can use context from both directions.

**Why:** The backward network needs information from the right-hand side of the sequence. If those inputs have not arrived yet, the model must wait or operate on a limited chunk of the sequence.

##  BiRNN vs. BiLSTM vs. BiGRU

These names describe the recurrent cell wrapped by the bidirectional structure:

| Name | Recurrent cell | Direction of processing |
|---|---|---|
| BiRNN (as used here) | Usually a simple/vanilla RNN | Forward and backward |
| BiLSTM | LSTM | Forward and backward |
| BiGRU | GRU | Forward and backward |

The bidirectional idea is the same in all three: process the sequence in both directions and combine the two representations. The difference is the recurrent cell used inside each direction.

##  Beginner-friendly worked example

Suppose the input is:

`I | live | near | Amazon | river`

For the word **Amazon**:

- The forward network has processed `I live near Amazon`.
- The backward network has processed the sequence from the end toward `Amazon`, using the context on its right.
- The model combines both states for the `Amazon` position.
- The output layer uses the combined representation to predict a label.

This can help the model use both left context (*live near*) and right context (*river*) when deciding what **Amazon** refers to.

##  Key takeaways

- A unidirectional RNN processes a sequence in one direction.
- A bidirectional RNN uses a forward network and a backward network.
- Their hidden states are combined at each sequence position, commonly by concatenation.
- BiLSTM and BiGRU apply the same bidirectional idea to LSTM and GRU cells.
- Bidirectional models can help with context-sensitive tasks such as NER and POS tagging.
- The extra direction increases parameter count and can increase computation.
- Bidirectional processing may add latency when the full input sequence is not immediately available.
- For forecasting, ensure that the model does not use information that would be unavailable at prediction time.



