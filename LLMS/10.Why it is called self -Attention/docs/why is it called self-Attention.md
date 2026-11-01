# Self-Attention Explained


### Question 1: What problem exists in traditional Sequence-to-Sequence (Seq2Seq) models?
**Answer:**
Traditional Seq2Seq models (like standard Encoder-Decoder LSTMs used for machine translation) process input sentences word by word and compress the entire input sequence into a single, fixed-length context vector. For longer sentences (especially those exceeding 30 words), forcing all semantic meaning into one fixed-size vector creates an information bottleneck, leading to information loss and degraded output quality .
---

### Question 2: How does the Luong Attention mechanism solve this bottleneck?
**Answer:**
Instead of relying on a single context vector for the entire sentence, Luong attention generates a dedicated context vector for every time step in the decoder. It calculates alignment scores between the current decoder state and every encoder state, normalizes those scores into probabilities using Softmax, and computes a weighted sum over all encoder states. This allows the model to focus on specific parts of the input sequence when generating each output word.
---

### Question 3: Why is Self-Attention called "Attention"?
**Answer:**
Even though Self-Attention does not use separate encoder and decoder networks, it follows the exact same three-step process as traditional attention mechanisms like Luong attention:
1. **Query (Asking for information):** The Query vector represents what context a word is seeking, playing the role of the decoder hidden state .
2. **Key (Matching similarity):** The Key vector acts like an encoder state label to calculate similarity scores between words.
3. **Value (Weighting content):** Softmax turns similarity scores into normalized weights, which are multiplied by Value vectors (the content representations) to form the output.

Because it relies on the same sequence of dot-product similarity, Softmax normalization, and weighted summation, it is fundamentally an attention mechanism.
---

### Question 4: Why is it called "Self" Attention?
**Answer:**
Traditional attention mechanisms (such as Luong or Bahdanau attention) compute attention between **two separate sequences**—for example, comparing an English source sentence in the encoder against a Hindi target sentence in the decoder (Inter-Sequence Attention).

Self-Attention computes attention between words **within the exact same single sequence** (Intra-Sequence Attention), calculating how each word in a sentence relates to every other word in that same sentence. Because the sequence pays attention to itself, it is named "Self-Attention".
---

### Quick Comparison Table

| Concept | Traditional (Luong) Attention | Self-Attention |
| :--- | :--- | :--- |
| **Sequence Interaction** | Inter-sequence (between two different sequences) | Intra-sequence (within the same sequence) |
| **Request Vector** | Decoder Hidden State | Query Vector |
| **Matching Label** | Encoder Hidden State | Key Vector |
| **Content Representation** | Encoder Hidden State  | Value Vector  |
