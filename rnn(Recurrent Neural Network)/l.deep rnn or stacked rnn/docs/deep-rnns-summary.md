# Deep RNNs or Stacked RNN

##  What Is a Deep RNN?

A **Deep RNN** (also called a **Stacked RNN**) is a recurrent neural network that has multiple recurrent hidden layers stacked vertically.

- A normal or single-layer RNN has one recurrent hidden layer.
- A Deep RNN passes information through multiple recurrent layers.
- Each layer processes information across the sequence's time steps.
- The output of one recurrent layer becomes the input to the next recurrent layer.

The idea is similar to adding more hidden layers to a basic feed-forward neural network: extra layers can help the model learn more complex representations of the data.

### Simple mental picture

Imagine analysing a product review:

1. The first recurrent layer detects useful word-level clues, such as *love*, *hate*, *amazing*, or *terrible*.
2. A middle layer can combine clues into phrase- or sentence-level meaning, such as “audio is bad.”
3. A higher layer can combine information across the review to estimate its overall sentiment.

This hierarchy is an illustrative way to understand the motivation for stacking layers; the model is not explicitly instructed to assign each layer one of these exact jobs.

---

##  Review: How a Normal RNN Processes a Sequence

Consider a sentiment-analysis task. The input review contains a sequence of words, and the goal is to predict whether the review is positive or negative.

At each time step, an RNN receives:

- The current input word, represented numerically.
- The hidden state from the previous time step.

It combines this information to calculate a new hidden state. That state carries information forward to the next time step.

For a review with five words, the RNN is unfolded into five time steps. For a three-word review, it has three time steps.

### Key terms

- **Time step:** The position of an item in the sequence.
- **Hidden state:** The information carried by the RNN from one time step to the next.
- **Unfolding through time:** Drawing the same recurrent cell at each time step to show how information moves through a sequence.
- **Feedback/recurrent connection:** The connection that carries the previous hidden state forward.

The same recurrent cell's parameters are reused across the time steps of a sequence.

---

##  How Stacking RNN Layers Works

A Deep RNN introduces a second direction of processing: **depth**.

- **Time axis (`t`):** Moves along the sequence—first word, second word, third word, and so on.
- **Layer/depth axis (`l`):** Moves vertically through the recurrent layers.

For example, a two-layer RNN processes each time step as follows:

1. The first recurrent layer receives the current word and its own previous hidden state.
2. It calculates its hidden representation for that time step.
3. The second recurrent layer receives the first layer's representation at the same time step, along with its own previous hidden state.
4. The second layer calculates its hidden representation.
5. The model produces the final prediction from the appropriate output representation.

This pattern continues at every time step. Each recurrent layer has its own recurrent connection across time.

### Important distinction

Stacking layers is **not** the same as adding more time steps.

- More time steps mean a longer input sequence.
- More layers mean greater network depth and additional transformations of the sequence representations.

A network can have many time steps and many recurrent layers at the same time.

---

##  Understanding the Architecture and Information Flow

The transcript illustrates a three-word sequence (`cat`, `mat`, `rat`) and a two-layer RNN.

For each word:

- The word is represented numerically and enters the first recurrent layer.
- The first layer combines it with the state from its previous time step.
- The first layer's current representation is passed to the second layer.
- The second layer combines that representation with its own previous state.
- After the sequence has been processed, the model can use the final representation to predict sentiment.

The transcript's illustration uses three units in the first recurrent layer and two units in the second. The number of units can differ between layers.

### What the connections represent

- **Input-to-hidden connection:** Transforms the current input into a hidden representation.
- **Recurrent connection:** Carries information from the previous time step within the same layer.
- **Layer-to-layer connection:** Passes the current representation from one recurrent layer to the next.
- **Output connection:** Uses a hidden representation to calculate the prediction.

For binary sentiment classification, the example uses a final output unit with a sigmoid activation, producing a value that can be interpreted as the model's predicted positive-class probability.

---

##  Notation for a Deep RNN

A Deep RNN can be drawn as a two-dimensional grid.

- `t` identifies the time step.
- `l` identifies the recurrent layer.
- `h_t^l` denotes the hidden state at time step `t` in layer `l`.

A hidden state depends on two things:

1. The hidden state from the **previous time step in the same layer**, `h_(t-1)^l`.
2. The representation from the **previous layer at the same time step**, `h_t^(l-1)`.

For a standard tanh-based RNN cell, the transcript gives the general equation:

\[
h_t^l = \tanh\left(W^l h_{t-1}^l + U^l h_t^{l-1} + b^l\right)
\]

Where:

- `h_t^l`: Current hidden state at time `t`, layer `l`.
- `W^l`: Recurrent weight matrix for layer `l`.
- `h_(t-1)^l`: Previous hidden state in the same layer.
- `U^l`: Weight matrix connecting the previous layer to the current layer.
- `h_t^(l-1)`: Current-time-step representation from the layer below.
- `b^l`: Bias for layer `l`.
- `tanh`: Activation function used by the RNN cell in this example.

**How to read the equation:** The cell combines the previous state and the current input representation using learned weights, adds a bias, and applies `tanh` to produce the current hidden state.

The exact shapes of the weight matrices depend on the number of units in the connected layers.

---

##  Why Use Deep RNNs?

### A. Hierarchical representations

Many sequences contain information at different levels.

In a product review:

- **Word level:** Words such as *love*, *hate*, *amazing*, and *terrible* can provide basic clues.
- **Phrase or sentence level:** A phrase such as “audio is bad” combines words into a more meaningful clue.
- **Whole-review level:** A review such as “Audio is bad, but the display is great. Overall, I am happy” requires combining different pieces of evidence.

Stacked recurrent layers can learn progressively richer representations that may help capture these levels of information.

### B. Advanced architectures

Deep recurrent layers can be used as components of more advanced models, such as:

- Encoder–decoder architectures.
- Sequence-to-sequence systems.
- Machine-translation systems.
- Architectures that use attention mechanisms.

The transcript discusses Deep RNNs in the context of encoder–decoder models with attention for machine translation.

---

## When Should You Try a Deep RNN?

The transcript highlights four practical situations:

1. **The problem is complex.** Examples include speech recognition and machine translation.
2. **You have enough training data.** Deep models can overfit when the dataset is too small.
3. **You have sufficient computational resources.** More parameters and more computation can increase training time.
4. **A simpler baseline is not good enough.** If a single-layer recurrent model gives unsatisfactory results, a deeper model is one option to test.

These are guidelines rather than guarantees. A deeper model does not automatically perform better; it should be evaluated on held-out data.

---

##  Building a Deep RNN in Keras

The transcript demonstrates a sentiment-classification model using the built-in IMDB dataset in Keras.

### Workflow shown in the transcript

1. Import the required libraries.
2. Load the IMDB dataset, limiting the vocabulary to 10,000 words.
3. Split the data into training and testing sets.
4. Pad or truncate reviews to a maximum length of 100 tokens.
5. Use an embedding layer to represent each token as a 32-dimensional vector.
6. Add a first `SimpleRNN` layer with 5 units.
7. Add a second `SimpleRNN` layer with 5 units.
8. Add a final dense output unit with sigmoid activation for binary sentiment classification.
9. Compile and train the model.

### Model structure

```python
Embedding(input_dim=10000, output_dim=32, input_length=100)
SimpleRNN(5, return_sequences=True)
SimpleRNN(5)
Dense(1, activation="sigmoid")
```

This is a structural illustration of the layers described in the transcript; exact Keras arguments can vary with the Keras version.

### What the embedding layer does

- `input_dim=10000`: The vocabulary is capped at 10,000 token IDs.
- `output_dim=32`: Each token is represented by a learned vector of 32 numbers.
- `input_length=100`: The example uses sequences padded or truncated to 100 tokens.

An embedding layer learns a vector representation for each token ID. It does not mean that a review has only 32 words; **32 is the dimensionality of each token's vector**.

### Why `return_sequences=True` matters

When one recurrent layer feeds another recurrent layer, the next layer needs a representation for every time step.

- The first `SimpleRNN` therefore uses `return_sequences=True` so it returns a sequence of hidden states, not just one final state.
- The second `SimpleRNN` can return only its final hidden state when the final classifier needs one representation for the whole review.
- If a task needs an output at every time step, the final recurrent layer may also need to return sequences.

**Practical rule:** In a stacked recurrent model, intermediate recurrent layers normally return sequences so the next recurrent layer can process the full sequence.

---

## Parameter Counts in the Transcript's Example

The transcript calculates the following parameter counts for the example model:

| Layer | Parameter count |
|---|---:|
| Embedding: 10,000 tokens × 32 dimensions | 320,000 |
| First SimpleRNN: input size 32, 5 units | 190 |
| Second SimpleRNN: input size 5, 5 units | 55 |
| Dense output: 5 inputs, 1 output | 6 |
| **Total** | **320,251** |

### How the SimpleRNN count is calculated

For a standard SimpleRNN layer with input dimension `d` and `u` units, the parameter count is:

`(d × u) + (u × u) + u`

These terms correspond to:

- Input-to-hidden weights.
- Hidden-to-hidden recurrent weights.
- One bias per unit.

For the first recurrent layer:

- Input weights: `32 × 5 = 160`
- Recurrent weights: `5 × 5 = 25`
- Biases: `5`
- Total: `160 + 25 + 5 = 190`

For the second recurrent layer:

- Input weights: `5 × 5 = 25`
- Recurrent weights: `5 × 5 = 25`
- Biases: `5`
- Total: `25 + 25 + 5 = 55`

For the final dense layer:

- Weights: `5 × 1 = 5`
- Bias: `1`
- Total: `6`

The embedding layer has `10,000 × 32 = 320,000` learned embedding values.

---

##  Deep RNNs, Deep LSTMs, and Deep GRUs

The stacking concept is not limited to `SimpleRNN`.

- **Deep RNN:** Multiple recurrent layers stacked together.
- **Deep LSTM:** Multiple LSTM layers stacked together.
- **Deep GRU:** Multiple GRU layers stacked together.

The transcript recommends experimenting with stacked LSTM and GRU models, noting that these architectures are often used in practical deep recurrent models. It also points out that the vanishing- and exploding-gradient problems associated with recurrent networks can still be a concern in deep RNNs.

### Why LSTM and GRU may be preferable

LSTMs and GRUs use gating mechanisms to control information flow and are designed to help with learning longer-range dependencies. They do not eliminate every training difficulty, and performance depends on the task, data, and model setup.

---

##  Backpropagation Through a Deep RNN

In a Deep RNN, the architecture forms a grid across both time and depth.

During training, gradients can flow through:

- Recurrent connections across time steps.
- Connections between recurrent layers.
- Paths that combine both directions.

This makes gradient calculations more involved than in a single recurrent layer. The transcript notes that libraries such as Keras handle the computation graph and gradient calculations internally.

The underlying learning process still relies on backpropagation and parameter updates; the deeper structure adds more paths through which gradients can flow.

---

##  Disadvantages and Practical Precautions

### A. Higher risk of overfitting

A deeper model has more parameters and may learn details specific to the training data rather than patterns that generalize.

Ways to reduce this risk include:

- Dropout.
- Regularization.
- Careful model design.
- Evaluating on validation and test data.
- Tuning the learning rate and weight initialization.

### B. Longer training time

More layers generally mean more computation and more parameters to update. Training can take longer, especially with large datasets or limited hardware.

### C. Gradient difficulties

Vanishing and exploding gradients can remain a concern in deep recurrent networks. LSTM and GRU layers may help with some recurrent-learning challenges, but they are not a guarantee against all gradient problems.

---

##  Practical Experiment Plan

To understand whether depth helps on a real task, compare models using the same dataset and evaluation setup:

1. Train a single-layer SimpleRNN.
2. Train a stacked Deep RNN.
3. Train a single-layer LSTM and a stacked Deep LSTM.
4. Train a single-layer GRU and a stacked Deep GRU.
5. Record training time, training performance, validation performance, and test performance.
6. Compare the results rather than assuming that the deepest model is best.

Keep preprocessing and train/validation/test splits consistent so that the comparison is meaningful.

---

##  Quick Revision

- **Deep RNN = Stacked RNN:** multiple recurrent hidden layers.
- **Time axis:** the sequence progresses from one time step to the next.
- **Depth axis:** representations pass from one recurrent layer to the next.
- **Hidden state:** carries information across time within a recurrent layer.
- **Layer-to-layer connection:** passes the current representation upward.
- **Hierarchical representations:** lower layers may learn simpler clues while higher layers can combine them into more complex representations.
- **`return_sequences=True`:** allows a recurrent layer to pass outputs for every time step to the next recurrent layer.
- **Deep LSTM and Deep GRU:** use the same stacking idea with LSTM or GRU cells.
- **Main disadvantages:** greater overfitting risk and longer training time; gradient difficulties may also arise.
- **Good practice:** compare against simpler baselines and evaluate on held-out data.



## Final Takeaway

A Deep RNN processes a sequence through multiple recurrent layers. Each layer combines information across time, while passing its representation to the next layer. This can help a model learn richer, hierarchical sequence representations, but it also increases computational cost and the risk of overfitting. Use deeper architectures when the problem, dataset, and available compute justify them—and verify their value through controlled experiments.
