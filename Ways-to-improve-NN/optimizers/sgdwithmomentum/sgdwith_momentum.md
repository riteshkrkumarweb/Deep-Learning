# Momentum in Deep Learning 🚀

## 1. What is Momentum?

Momentum is an **optimization technique** used with Gradient Descent to make weight updates **smoother and often faster**.

> **Momentum = Gradient Descent + memory of previous updates**

---

## 2. Why Do We Need Momentum?

Basic SGD uses only the current gradient.

This can cause:

* Zig-zag movement in narrow/high-curvature regions
* Slow progress in some directions
* Sensitivity to noisy mini-batch gradients

Momentum helps by using **previous updates along with the current gradient**.

---

## 3. Meaning of the Terms

* **$w$** → Weight/parameter being optimized
* **$v$** → Velocity/accumulated update
* **$t$** → Current iteration
* **$t-1$** → Previous iteration
* **$t+1$** → Next iteration
* **$\beta$** → Momentum/memory factor
* **$\eta$** → Learning rate
* **$L$** → Loss function
* **$\nabla L$** → Gradient of the loss

---

## 4. What is β?

$\beta$ controls **how much previous movement is remembered**.

For example, $\beta=0.9$ means a large portion of the previous movement is retained.

* Higher $\beta$ → More memory
* Lower $\beta$ → Less memory
* $\beta=0$ → No previous movement is remembered

---

## 5. Decay Factor

The same $\beta$ causes the influence of older updates to **decay over time**.

For example, with $\beta=0.9$:

* Recent movement → Strong influence
* Older movement → Weaker influence
* Very old movement → Much weaker influence

> **Momentum remembers the past, but the influence of older movements gradually becomes weaker.**

This is called **exponential decay**.

---

## 6. Intuition 🏀

Imagine a ball rolling downhill.

### SGD

The ball mainly reacts to the **current slope**.

### Momentum

The ball has **inertia from its previous movement**.

If several updates point in the same direction, Momentum builds up movement in that direction.

---

## 7. Advantages of Momentum

* **Faster convergence** → Can accelerate movement in consistent directions
* **Smoother updates** → Reduces the effect of noisy gradients
* **Less oscillation** → Particularly useful in steep/high-curvature directions
* **Better optimization movement** → Helps move through difficult loss surfaces

---

## 8. Disadvantages of Momentum

* **Overshooting** → Accumulated movement can pass the minimum
* **Oscillation** → Excessive momentum can cause back-and-forth movement
* **Hyperparameter tuning** → A suitable $\beta$ is important
* **Can become unstable** → Especially with an aggressive learning rate and momentum
* **Extra optimizer state** → The optimizer must maintain the velocity

---

## 9. Momentum vs SGD

| SGD | Momentum |
|---|---|
| Uses current gradient | Uses current gradient + previous updates |
| No accumulated velocity | Maintains velocity |
| Can zig-zag | Usually smoother |
| Can be slower | Often faster |
| Simpler | Slightly more complex |

---

## 10. Momentum vs Physical Momentum

In physics, momentum depends on **mass and velocity**.

In deep learning, the term **velocity** is used by analogy.

The optimization velocity is a **mathematical accumulated update**, not physical velocity.

There is **no physical mass** involved in Momentum Optimization.

---

## 🔑 Final Takeaway

> **Momentum remembers previous weight-update directions, combines them with the current gradient, and uses this accumulated movement to update the weights.**

### Easy Memory Trick

**Weight → What we change**

**Velocity → How we move**

**β → How much we remember**

**η → Learning rate**

**Gradient → Current direction/slope**

**Decay → Older information gradually becomes weaker**