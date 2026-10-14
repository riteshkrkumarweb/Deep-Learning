# Gated Recurrent Unit (GRU) 

## What Is a GRU?

A **Gated Recurrent Unit (GRU)** is a type of **Recurrent Neural Network (RNN) architecture** used to process **sequential data**.

Sequential data is data in which order matters. Examples include:
- Words in a sentence
- Time-series measurements
- A sequence of events
- Text reviews, where earlier words help interpret later words

A GRU processes a sequence one time step at a time. At each time step, it receives:
1. The **previous hidden state**, which carries information from earlier steps.
2. The **current input**, which represents the current word, token, or other item in the sequence.

Its goal is to calculate the **current hidden state** from the previous hidden state and the current input. The detailed calculation is explained in the architecture section.

##  Why Do We Need GRUs if LSTMs Already Exist?

### The problem with a simple RNN

A simple RNN can struggle to preserve information over long sequences. The transcript connects this limitation to:
- **Vanishing gradients**
- **Exploding gradients**

Because of these problems, a simple RNN may fail to retain useful long-term context.

###  How LSTM helps

A Long Short-Term Memory network (LSTM) was designed to improve the handling of long-term dependencies. In the transcript's explanation, an LSTM maintains two separate states:
- **Cell state:** a pathway for carrying information over time
- **Hidden state:** the state used to represent the current context and produce outputs

An LSTM has three main gates:
1. Forget gate
2. Input gate
3. Output gate

These gates control the flow of information.

### Why GRU was introduced

The transcript explains that GRUs were introduced later than LSTMs and aim to provide a simpler gated recurrent architecture.

A GRU:
- Uses **two gates** instead of the LSTM's three.
- Maintains **one hidden state**, rather than separate cell and hidden states.
- Usually has **fewer parameters** than an equivalent LSTM.
- Can therefore be faster to train or compute in some settings.
- Can perform comparably to an LSTM, and on some tasks may perform better.

**Important:** GRU is not always better than LSTM. The transcript recommends testing both on the actual dataset and task.

##  The Big Idea Behind GRU

The central idea is that a GRU does not need separate cell and hidden states to carry context. It uses a **single hidden state** to carry information forward.

At each time step, the GRU decides:
- Which parts of the previous hidden state should be reset or reduced for the current input.
- What a candidate updated state should look like.
- How to balance the previous state against the candidate state.

It uses two gates:
- **Reset gate** r_t
- **Update gate** z_t

The two gates regulate information flow through the hidden state.

##  GRU Notation and Terminology

| Symbol | Name | Meaning |
|---|---|---|
| x_t | Current input | Vector representing the input at time step t |
| h_{t-1} | Previous hidden state | Context or memory carried from the previous time step |
| r_t | Reset gate | Controls how much of the previous hidden state is used when creating the candidate state |
| tilde{h}_t | Candidate hidden state | A proposed new hidden state formed from the current input and reset-modulated past state |
| z_t | Update gate | Controls the balance between the previous state and candidate state |
| h_t | Current hidden state | The updated context passed to the next time step |

All of these are vectors. In the transcript's example, the hidden-state vectors have four components. In a real model, the hidden-state size is a design choice.

Except for x_t, the vectors listed above have the same dimension as the hidden state. If the hidden size is h, then h_{t-1}, r_t, tilde h_t, z_t, and h_t each have h components.

##  What Does the Hidden State Represent?

The hidden state is the GRU's running **memory or context**.

As a sequence is processed, the hidden state is updated to reflect information from the sequence so far. It is not a literal, human-readable summary. Its components are learned numerical features.

###  The lecture's story example

The transcript uses a fictional story about King Vikram, his enemy Kali, and later generations seeking revenge. To build intuition, it imagines four hidden-state dimensions representing:

1. Power
2. Conflict
3. Tragedy
4. Revenge

For example, a state with values 0.7, 0.8, 0.3, and 1.0 is **illustratively interpreted** as strong power-related context, high conflict, lower tragedy, and a strong revenge-related context.

These labels are only a teaching analogy. A trained GRU does **not** automatically assign named meanings such as “power” or “revenge” to individual dimensions. The model learns numerical representations, and their individual meanings are generally not directly interpretable.

###  How context changes across sentences

The lecture imagines the following sequence:
1. Vikram is introduced as powerful.
2. Kali is introduced as an enemy.
3. Kali fights and kills Vikram.
4. Vikram's son grows up and becomes strong.
5. The son attacks Kali but is killed.
6. A later descendant fights Kali.
7. The descendant kills Kali and takes revenge.

As each sentence arrives, the hidden state can change. The current input may add new information, make earlier information less relevant, or alter the interpretation of the story so far.

This illustrates the purpose of the gates: they help the model control how the hidden state changes at each time step.

##  Inputs Are Vectors, Not Raw Words

A neural network does not directly process a word such as `cat` as a human-readable word. Text must first be represented numerically.

The transcript demonstrates **one-hot encoding** with three unique words:

- `cat` → [1,0,0]
- `mat` → [0,1,0]
- `rat` → [0,0,1]

Consider these example sentences:
- `cat mat rat`
- `cat rat rat`
- `mat rat mat`

With one-hot encoding, each word becomes a vector. The model receives one word vector at each time step.

Other representations mentioned in the transcript include:
- Bag of Words
- Word2Vec
- Other embedding techniques

The input-vector size depends on the representation method. It does not have to equal the hidden-state size.

##  GRU Architecture: The Four Main Steps

The transcript divides the GRU calculation into four steps:

1. Calculate the **reset gate** r_t.
2. Calculate the **candidate hidden state** tilde h_t.
3. Calculate the **update gate** z_t.
4. Calculate the **current hidden state** h_t.

The data flow is:

```text
Previous hidden state h_(t-1) ──┬──> Reset gate r_t ──┐
                                │                     │
Current input x_t ──────────────┘                     ├─> Modulate old state
                                                      │
Modulated old state + current input ──────────────────┴─> Candidate state h~_t

Previous hidden state h_(t-1) + current input x_t ───────> Update gate z_t

Previous state h_(t-1) + candidate state h~_t + update gate z_t
                         └──────────────────────────────> Current state h_t
```

The following sections explain each step.

##  Step 1 — Reset Gate

###  What is the reset gate?

The reset gate r_t is a vector whose values are between 0 and 1 because it is produced by applying the sigmoid activation function to a learned weighted combination of the previous hidden state and current input, plus a bias.

Where:
- Sigmoid is the activation function used for the gate.
- The reset-gate weight matrix contains learned weights.
- Concatenation joins the previous hidden state and current input into one vector.
- The bias vector is a trainable value added during the transformation.

The reset gate uses both the previous hidden state and current input to decide how much of the past state to use when forming the candidate hidden state.

###  How to interpret gate values

A value close to 0 means that the corresponding component of the previous state is suppressed strongly when the reset-modulated state is formed.

A value close to 1 means that the corresponding component is mostly retained for that calculation.

The gate is **element-wise**: different hidden-state dimensions can receive different gate values.

###  The transcript's illustrative example

Suppose the previous state has four values: 0.6, 0.6, 0.7, and 0.1.

Imagine the dimensions are labelled power, conflict, tragedy, and revenge for explanation only. For a new sentence about Vikram's son becoming strong, the transcript imagines reset-gate values of 0.8, 0.2, 0.1, and 0.9. Multiplying each gate value by the matching previous-state value gives 0.48, 0.12, 0.07, and 0.09.

This is called the **reset-modulated hidden state**, **modulated memory**, or **reset memory** in the lecture.

###  What does “reset” mean?

It does not mean that the entire hidden state is necessarily erased. It means that the gate can reduce selected components of the previous state before the candidate state is calculated.

##  Element-Wise Operations

GRU uses element-wise operations, meaning the operation is applied separately to matching positions in two vectors.

###  Element-wise multiplication

Element-wise multiplication means multiplying values at matching positions in two vectors. For example, if one vector contains 2, 3, and 4 and another contains 5, 6, and 7, the result contains 10, 18, and 28.


###  Element-wise addition

Add matching positions in the two vectors. For example, adding 2, 3, and 4 to 5, 6, and 7 produces 7, 9, and 11.

###  Element-wise subtraction from one

For a gate vector with values 0.1, 0.7, 0.8, and 0.2, subtract each value from 1 to obtain 0.9, 0.3, 0.2, and 0.8.

The scalar 1 is applied to every component. These operations require compatible vector dimensions.

##  Step 2 — Candidate Hidden State

###  What is the candidate hidden state?

The **candidate hidden state** (shown as h with a tilde in the lecture) is a proposed new representation of the context at the current time step.

It is calculated using:
- The current input x_t
- The reset-modulated previous hidden state, created by multiplying each previous-state value by its matching reset-gate value

The GRU joins the reset-modulated previous hidden state with the current input, applies a learned weighted transformation and bias, and then applies tanh.

Where:
- The candidate-state weight matrix contains learned weights.
- The candidate-state bias vector contains trainable bias values.
- Tanh is the hyperbolic tangent activation function.
- The combined input contains the reset-modulated past state and the current input.

The output of tanh is between -1 and 1.

###  Why not use the candidate state directly?

The candidate state is strongly influenced by the current input and the reset-modulated past. However, the new input may not be important enough to replace the entire previous context.

Therefore, the GRU does not simply replace the current hidden state with the candidate hidden state. It uses the update gate to balance the candidate state with the previous hidden state.

###  Candidate-state example from the lecture

The transcript imagines that after processing a new sentence, the candidate state might contain 0.7, 0.2, 0.1, and 0.2.

This is an illustrative value, not the result of a complete numerical neural-network calculation. The lecture uses it to show that the candidate state proposes how the context could change.

##  Step 3 — Update Gate

###  What is the update gate?

The update gate z_t decides how to balance:
- The previous hidden state h_{t-1}
- The candidate hidden state tilde h_t

It is calculated by joining the previous hidden state and current input, applying a learned weighted transformation and bias, and then applying sigmoid. The update-gate weight matrix contains learned weights, and its bias vector contains trainable bias values. Sigmoid ensures each component of the update gate is between 0 and 1.

###  Interpreting the update gate

In the transcript's convention for the final-state calculation:

- A **high** update-gate value gives more weight to the candidate state.
- A **low** update-gate value gives more weight to the previous hidden state.

The gate works component by component, so it can retain some dimensions while updating others.

###  Practical analogy

Imagine reading a mystery story:
- If the current sentence reveals who committed the crime, it may substantially change your interpretation. The candidate state should receive more weight.
- If the current sentence is a minor detail, the earlier context may remain more important.

The update gate learns how to balance these sources of information during training.

##  Step 4 — Calculate the Current Hidden State

The transcript calculates the current hidden state by multiplying the previous hidden state by the complement of the update gate, multiplying the candidate hidden state by the update gate, and adding the two results element by element.

This is a weighted combination performed element by element.

###  What each part does

The previous-state contribution is calculated by multiplying each previous hidden-state value by one minus the matching update-gate value.

The candidate-state contribution is calculated by multiplying each candidate hidden-state value by the matching update-gate value.

Adding the two vectors produces the current hidden state h_t.

### Full numerical example from the transcript

Assume the previous state contains 0.6, 0.6, 0.7, and 0.1; the candidate state contains 0.7, 0.2, 0.1, and 0.2; and the update gate contains 0.1, 0.7, 0.8, and 0.2.

First, subtract each update-gate value from 1, giving 0.9, 0.3, 0.2, and 0.8. Multiply these values by the previous state to get 0.54, 0.18, 0.14, and 0.08.

Next, multiply the update-gate values by the corresponding candidate-state values to get 0.07, 0.14, 0.08, and 0.04. Add the two results position by position. The final hidden state is 0.61, 0.32, 0.22, and 0.12.

This is the new context passed forward to the next time step.

###  Why the update gate matters

The gate makes the update adaptive:
- If z_t is close to 0 for a component, that component mostly preserves the previous state.
- If z_t is close to 1 for a component, that component mostly uses the candidate state.
- Intermediate values blend the two.

##  GRU Process — Quick Reference

The operations described in the transcript can be remembered in this order:

1. **Reset gate:** Combine the previous hidden state and current input, apply a learned weighted transformation and bias, then use sigmoid to produce gate values.
2. **Reset-modulated previous state:** Multiply each previous-state component by the matching reset-gate value.
3. **Candidate hidden state:** Combine the reset-modulated state with the current input, apply a learned weighted transformation and bias, then use tanh.
4. **Update gate:** Combine the previous hidden state and current input, apply a learned weighted transformation and bias, then use sigmoid.
5. **Current hidden state:** Blend the previous state and candidate state, using the update gate to decide the contribution of each.

### Important terminology

- **Sigmoid:** Activation function that produces values between 0 and 1.
- **Tanh:** Hyperbolic tangent activation function that produces values between -1 and 1.
- **Weight matrices:** Trainable values used to transform input vectors.
- **Bias vectors:** Trainable values added during a transformation.
- **Concatenation:** Joining vectors into one longer vector.
- **Element-wise multiplication:** Multiplying corresponding positions in two vectors.
- **Previous hidden state:** Context carried from the previous time step.
- **Current hidden state:** Updated context passed forward.
- **Candidate hidden state:** Proposed new context.

##  Dense Layers, Hidden Size, and Parameter Shapes

The transcript explains the gate calculations as fully connected neural-network layers.

Let:
- d means the input-vector dimension.
- h means the hidden-state dimension.

Concatenating h_{t-1} and x_t gives a vector of size h+d.

For each gate or candidate layer in the transcript's formulation:
- Weight matrix shape: the combined input size multiplied by the hidden size, using the lecture's row-vector convention.
- Bias vector size: h.
- Output vector size: h.

For example, if the hidden-state size is 4 and the input-vector size is 3:
- The concatenated input has 7 values (4 hidden-state values plus 3 input values).
- Each layer produces 4 values.
- Each corresponding weight matrix has 28 weights (7 input positions connected to 4 output units).
- Each bias vector has 4 values.

The transcript uses these dimensions to explain how the reset gate is computed. The same input and output dimensions apply to the update-gate and candidate-state layers in this simplified formulation.

### Parameter count under this formulation

Each of the three transformations (reset gate, update gate, and candidate state) has one weight for each connection from the combined input to each hidden unit, plus one bias for each hidden unit. The total parameter count is three times the number of weights and biases in one transformation. This count includes the input and recurrent weights combined in each transformation and assumes one bias vector per transformation.

##  GRU vs. LSTM

| Feature | GRU | LSTM |
|---|---|---|
| Main gates in the lecture | Reset and update | Forget, input, and output |
| State variables | One hidden state | Hidden state and cell state |
| Architecture | Simpler | More complex |
| Parameter count | Usually lower for comparable dimensions | Usually higher for comparable dimensions |
| Training/computation | Can be faster | Can require more computation |
| Performance | Can match or outperform LSTM on some tasks | Can outperform GRU on other tasks |
| Choosing between them | Evaluate on the actual task | Evaluate on the actual task |

### Practical rule from the lecture

- Try a GRU when you want a simpler model with fewer parameters.
- Try an LSTM if the GRU's results are not good enough.
- Compare the models empirically on the same data, with fair training and evaluation settings.

Neither architecture is universally best.

##  Common Confusions

### 16.1 Is the candidate hidden state the current hidden state?

No.

- tilde h_t is the proposed state.
- h_t is the final state after balancing the candidate state with h_{t-1} using z_t.

###  Does the reset gate directly calculate the final hidden state?

No. It modulates the previous hidden state before the candidate hidden state is calculated.

###  Does the update gate reset the previous state?

Not exactly. The update gate controls the balance between the previous hidden state and the candidate state. The reset gate affects how the previous state contributes to the candidate-state calculation.

###  Are gate values the same as learned weights?

No. The weights and biases are trainable parameters. Gate values are the outputs calculated from those parameters, the previous hidden state, and the current input for a particular time step.

###  Are the hidden-state dimensions human-readable concepts?

Not necessarily. The lecture's “power, conflict, tragedy, revenge” dimensions are a teaching analogy. In an actual model, hidden-state dimensions are learned numerical features.

###  Does GRU always train faster or perform better than LSTM?

No. Fewer parameters can reduce computation, but actual speed and accuracy depend on the task, data, implementation, hardware, and training settings.


##  Key Takeaways

1. **GRU is a gated RNN architecture** for sequential data.
2. It carries context using one hidden state.
3. The **reset gate** controls how much of the previous state is used when forming the candidate state.
4. The **candidate hidden state** proposes a new representation based on the current input and reset-modulated past.
5. The **update gate** controls how the previous state and candidate state are combined.
6. The final state is created by blending the previous hidden state and candidate hidden state. The update gate controls the contribution of each, element by element.

7. GRUs often have fewer parameters than comparable LSTMs, but performance depends on the task.
8. A practical comparison should test both architectures under the same conditions.

---

## 19. Clarifications and Extra Context Beyond the Transcript

This section separates brief clarifications from the lecture-derived explanation.

### A. Sigmoid's exact range

The sigmoid function produces values strictly between 0 and 1, though outputs can be very close to either endpoint. This is why it is useful for gates.

### B. Tanh's exact range

The tanh activation produces values strictly between -1 and 1. In the transcript's GRU explanation, it is used when generating the candidate hidden state.

### C. Parameter-count comparison caveat

The lecture describes three transformations for the GRU: reset gate, update gate, and candidate hidden state. Exact parameter counts can differ across frameworks and implementations, including how biases are represented and whether separate input and recurrent bias vectors are used. When checking a model's actual parameter count, inspect the specific implementation.

### D. Gate convention caveat

These notes preserve the transcript's convention: a high update-gate value gives more weight to the candidate state, while a low value gives more weight to the previous state. Some references and software implementations define the update gate with the opposite convention. If a reference uses a different convention, check its definition before comparing descriptions.

### E. A hidden state is not a guaranteed readable summary

It is useful to think of the hidden state as the network's running context, but its coordinates do not necessarily correspond to neat human concepts. The story analogy is only for intuition.

### F. The transcript's numerical story values are illustrative

The values assigned to power, conflict, tragedy, and revenge are invented to make the mechanism easier to understand. They are not outputs from a trained GRU model, and they do not show the model's actual learned internal features.
