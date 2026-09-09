# ReLU & Its Variants — Quick Summary

## Dying ReLU

**Definition:** Dying ReLU is a problem where a ReLU neuron becomes permanently inactive because it keeps receiving negative inputs, producing `0` output and a `0` gradient.

**Why does it happen?**

`Negative input → ReLU output = 0 → Gradient = 0 → Weight cannot update → Neuron stops learning`

So the neuron is called **"dead"** because it can no longer learn.

**Example:**

If a neuron keeps receiving:

`-5, -3, -8, -2`

ReLU produces:

`0, 0, 0, 0`

Its gradient remains `0`, so the neuron may stop updating.

**Solution:** Use **Leaky ReLU, PReLU, ELU**, etc.

---

## ReLU

`f(x) = max(0, x)`

**Advantages**

* Simple and fast
* Non-saturating for positive inputs
* Good gradient flow for positive inputs

**Disadvantages**

* Dying ReLU problem
* Not zero-centered

**Use:** General hidden layers; common default.

---

## Leaky ReLU

Allows a small negative output instead of `0`.

**Advantages**

* Reduces Dying ReLU
* Gradient remains non-zero for negative inputs
* Fast

**Disadvantages**

* Negative slope is fixed
* Not zero-centered

**Use:** When you want a simple solution to Dying ReLU.

---

## PReLU

Like Leaky ReLU, but the negative slope `α` is **learned**.

**Advantages**

* Reduces Dying ReLU
* Learns the best negative slope

**Disadvantages**

* Adds learnable parameters
* Can potentially overfit

**Use:** When ReLU causes problems and you want the slope to be learned.

---

## ELU

Uses ReLU-like behavior for positive values and a smooth negative curve.

**Advantages**

* Reduces Dying ReLU
* Produces negative outputs
* More zero-centered than ReLU

**Disadvantages**

* More computationally expensive
* Saturates for very negative inputs

**Use:** When smoother negative-side behavior is useful.

---

## SELU

**Scaled ELU**

`SELU = λ × ELU`

The scaling helps activations maintain a **stable mean and variance**.

**Advantages**

* Encourages self-normalizing behavior
* Helps maintain stable activations

**Disadvantages**

* Requires specific conditions/setup
* Not a universal replacement for ReLU

**Use:** Self-normalizing neural networks.

**Scaled version:** Original function × scaling factor.

**Self-normalizing:** Activations naturally tend to maintain a stable mean and variance across layers.

---

## Softplus

Smooth approximation of ReLU.

`f(x) = log(1 + eˣ)`

**Advantages**

* Smooth
* No traditional Dying ReLU problem

**Disadvantages**

* More computationally expensive
* Can have very small gradients for large negative inputs

**Use:** When smoothness is important.

---

## GELU

A smooth activation widely used in modern architectures.

**Advantages**

* Smooth
* Works well in many modern deep-learning models

**Disadvantages**

* More computationally expensive than ReLU
* More complex

**Use:** **Transformers** and modern architectures.

---

## Mish

A smooth, non-monotonic activation.

**Advantages**

* Smooth
* Allows negative outputs

**Disadvantages**

* More computationally expensive
* More complex than ReLU

**Use:** Mostly experimentation and some modern architectures.

---

## Saturated vs Non-Saturated

**Saturated:** Gradient becomes very small → learning becomes slow.

Example: **Sigmoid** at very large positive/negative inputs.

**Non-saturated:** Gradient remains useful → learning can continue effectively.

Example: **ReLU for positive inputs**.

**Memory:**

`Saturated → flat → tiny gradient → slow learning`

`Non-saturated → useful gradient → better learning`

---

## Quick Selection

* **General hidden layers → ReLU**
* **Dying ReLU → Leaky ReLU**
* **Learn negative slope → PReLU**
* **Smooth negative side → ELU**
* **Self-normalizing network → SELU**
* **Transformers → GELU**
* **Need smooth ReLU-like function → Softplus**
* **Experimenting with smooth activation → Mish**
