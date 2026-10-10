# LSTM Architecture — Summary and Important Questions

## 1. What Is an LSTM?

**Long Short-Term Memory (LSTM)** is a type of recurrent neural network (RNN) designed to process sequential data while helping preserve useful information over longer sequences.

Compared with a basic RNN, an LSTM maintains two states:

- **Cell state:** the internal memory pathway that carries information across time steps.
- **Hidden state:** a numerical representation of information processed up to the current time step. It is also used as the output of the LSTM cell at that step.

Neither state is a literal copy of the words or a perfect record of every earlier detail.

## 2. What Happens at Each Time Step?

At each time step, an LSTM receives **three inputs**:

1. Previous cell state
2. Previous hidden state
3. Current input

It produces **two outputs**:

1. Updated cell state
2. Current hidden state

The new states are passed to the next time step.

### Example: “I love you”

- **Step 1 — “I”:** the LSTM processes “I” and produces the first hidden state and first cell state.
- **Step 2 — “love”:** it processes “love” together with the previous hidden state and previous cell state, producing updated states that reflect processing up to “I love.”
- **Step 3 — “you”:** it processes “you” together with the previous states, producing updated states that reflect processing up to “I love you.”

The states represent information processed so far; they do not literally store the sentence as text.

**Time step:** one position or iteration in a sequence. In this word-level example, each word is one time step.

## 3. The Three Gates and the Candidate

An LSTM has three main gates. The candidate cell state is an additional calculation, **not a fourth gate**.

| Component | Main job | Simple meaning |
|---|---|---|
| Forget gate | Controls how much of the previous cell state is retained | What should remain? |
| Input gate | Controls how much candidate information is added to memory | How much should be added? |
| Candidate cell state | Proposes new information for the cell state | Potential new information |
| Output gate | Controls how much of the updated memory contributes to the hidden state | What should be exposed? |

### Forget gate

The forget gate looks at the current input and previous hidden state to calculate values between zero and one. These values control how much of each corresponding component of the previous cell state remains.

- A value of **one** means retain all of that component.
- A value of **zero point five** means retain half of that component.
- A value of **zero** means remove that component.

These are element-wise operations on vectors, not decisions about whole words.

### Candidate cell state

**Candidate = proposal / potential.**

The candidate cell state is **proposed new information**, not the current cell state and not the previous cell state. It is calculated from the current input and previous hidden state.

The input gate controls how much of this proposed information is accepted into the cell state.

### Input gate

The input gate produces values between zero and one. It filters the candidate information, controlling how much of each candidate component is added to memory.

### Updating the cell state

The LSTM updates its memory in two parts:

1. Retain some information from the previous cell state, as controlled by the forget gate.
2. Add some candidate information, as controlled by the input gate.

The combination becomes the **updated cell state**.

### Output gate and hidden state

The updated cell state is first passed through the hyperbolic tangent function, often written as `tanh`. The output gate then controls which parts contribute to the hidden state.

The hidden state is sent to the next time step and can also be used for a prediction at the current step.

## 4. What the Components Do Mathematically

The gates and candidate cell state are calculated using the current input and previous hidden state. The learned weights and biases help the LSTM learn how to process this information.

- **Sigmoid function:** produces values between zero and one, making it useful for gates.
- **Tanh function:** produces values between negative one and positive one; it is used to create candidate information and to transform the updated cell state before producing the hidden state.
- **Weights and biases:** learned numerical values used in the calculations.
- **Concatenation:** joining the previous hidden state and current input into one vector.
- **Element-wise multiplication:** multiplying corresponding components of two vectors.
- **Element-wise addition:** adding corresponding components of two vectors.

## 5. Vector Shapes and Operations

The transcript uses an example where:

- The previous hidden state has three values.
- The current input has four values.
- Joining them gives a vector with seven values.
- Each gate and candidate layer has three units, so its output has three values, matching the cell-state and hidden-state dimensions.

These dimensions are an illustrative example, not a universal requirement.

Examples of element-wise operations:

- Multiplication: `[4, 5, 6]` and `[1, 2, 3]` produce `[4, 10, 18]`.
- Addition: `[4, 5, 6]` and `[1, 2, 3]` produce `[5, 7, 9]`.

## 6. How LSTM Helps With Long-Term Dependencies

A basic RNN can struggle to preserve learning signals across many time steps. This is associated with the **vanishing gradient problem**.

The LSTM cell-state pathway and gates provide a route for useful information and gradients to travel across time. For example, if the forget gate retains all of the previous cell state and the input gate adds nothing, the cell state remains unchanged for that step.

This illustrates how the cell state can remain unchanged in that particular case. It does **not** mean every LSTM always preserves all information perfectly or that vanishing gradients are impossible in every situation.

## 7. The Mobile Factory Analogy

This analogy can help build intuition, as long as it is not taken literally:

- **Current input:** the information arriving at the current step.
- **Cell state:** the evolving internal memory pathway.
- **Hidden state:** the current numerical representation of information processed so far.
- **Forget gate:** controls which components of existing memory are retained.
- **Candidate cell state:** proposes numerical information that could be added.
- **Input gate:** controls how much proposed information is added.
- **Output gate:** controls how much of the updated memory is exposed through the hidden state.

An LSTM does not consciously select entire “premium phones” or store literal objects. Its gates operate mathematically on vector components.

## 8. Important Questions From Our Chat

### Q1. What is a time step?

A time step is one position or iteration in a sequence. For the word-level sequence “I love you,” the words “I,” “love,” and “you” are three time steps.

### Q2. What is the hidden state?

The hidden state is a **numerical representation of the information processed by the LSTM from the beginning of the sequence up to the current time step**. It is passed forward and can be used for prediction.

### Q3. What is the cell state?

The cell state is the LSTM’s internal memory pathway that carries useful information across time steps. Gates control how much previous information is retained and how much new candidate information is added.

### Q4. Does the hidden state process all information while the cell state keeps only important information?

That is a useful first approximation, but it is not exact. Both states are involved in processing the sequence. The cell state provides a memory pathway, while the hidden state is the representation exposed at the current step. Neither is a literal full transcript, and neither guarantees that every detail is retained.

### Q5. Is the candidate cell state the current cell state?

**No.**

- **Previous cell state:** existing memory.
- **Candidate cell state:** proposed new information.
- **Current cell state:** updated memory after combining retained old information with accepted candidate information.

### Q6. Candidate in one word?

**Proposal** or **potential**.

### Q7. Does the LSTM receive all previous words again at every time step?

No. It receives the current input plus the previous hidden state and previous cell state. Information from earlier steps can be carried forward through these states.

## 9. Quick Revision

Remember the sequence:

1. **Forget gate** — controls retention of previous memory.
2. **Candidate cell state** — proposes new information.
3. **Input gate** — controls how much candidate information is added.
4. **Cell-state update** — combines retained memory and accepted new information.
5. **Output gate** — controls what part of the updated memory contributes to the hidden state.

**One-line summary:** An LSTM uses gates to control the retention, addition, and exposure of information as it processes a sequence one time step at a time.
