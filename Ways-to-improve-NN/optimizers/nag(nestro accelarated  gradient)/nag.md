# Nesterov Accelerated Gradient (NAG)

## 1. What is NAG?

**Nesterov Accelerated Gradient (NAG)** is a momentum-based optimization algorithm that improves upon Momentum by **looking ahead in the direction of accumulated (increased little by little over a period of time) momentum before calculating the gradient**.

> **NAG = Look Ahead → Calculate Gradient → Update**

---

## 2. Meaning of NAG

◇ **Nesterov** → Named after mathematician **Yurii Nesterov**

◇ **Accelerated** → Designed to improve the convergence of optimization

◇ **Gradient** → Uses the gradient of the loss function to determine the update direction

---

## 3. What is Look-Ahead?

**Look-ahead** means temporarily considering the position where the accumulated momentum is expected to take the model and calculating the gradient at that position before making the actual update.

> **Look-ahead = Check the gradient at the expected future position.**

---

## 4. Why NAG?

In standard Momentum, the gradient is calculated at the **current position**.

**Momentum:**

Current Position → Calculate Gradient → Combine with Momentum → Update Weights

**NAG:**

Current Position → Look Ahead using Momentum → Calculate Gradient → Adjust Momentum → Update Weights

NAG uses information about the slope **ahead of the current position**, which can lead to more informed updates.

---

## 5. Momentum vs NAG

### Momentum

> Calculates the gradient at the **current position**.

**Current Position → Gradient → Update**

### NAG

> Looks ahead in the direction of momentum and calculates the gradient at the **look-ahead position**.

**Current Position → Look Ahead → Gradient → Update**

### Key Difference

> **Momentum reacts to the current position, while NAG looks ahead before making the update.**

---

## 6. How NAG Works

### Step 1 — Start with Current Weights

The optimizer starts from the current weights.

### Step 2 — Look Ahead

The accumulated momentum is used to determine a temporary look-ahead position.

### Step 3 — Calculate the Gradient

The gradient is calculated at the **look-ahead position**, not the current position.

### Step 4 — Update Momentum

The previous momentum is combined with the newly calculated gradient.

### Step 5 — Update Weights

The new momentum is used to make the actual weight update.

### Step 6 — Repeat

The process continues until the optimization converges.

---

## 7. Geometric Intuition

The gradient tells us the direction of **steepest increase** in the loss.

Since Gradient Descent wants to reduce the loss, it moves in the **opposite direction of the gradient**.

NAG first looks ahead in the direction suggested by momentum and then checks the gradient there.

This gives the optimizer a better idea of what is happening **ahead**.

---

## 8. Overshooting

Momentum can build up a large velocity.

Because of this, the optimizer may move past the minimum.

This is called **overshooting**.

NAG can help control overshooting by checking the gradient at the look-ahead position before making the actual update.

---

## 9. Oscillation

**Oscillation** means repeatedly moving from one side of the minimum to the other.

With suitable hyperparameters, the movement can become smaller over time and eventually converge.

NAG can help reduce unnecessary oscillation, but it **does not completely eliminate it**.

---

## 10. Local Minimum

A **local minimum** is a point that is lower than the nearby points but may not be the lowest point in the entire loss landscape.

NAG can still settle at a local minimum.

---

## 11. Why NAG Does Not Guarantee the Global Minimum

NAG is still a **gradient-based optimization algorithm**.

It improves how the optimizer moves, but it does not know where the global minimum is.

A complex loss landscape can contain:

◇ Multiple local minima

◇ Saddle points

◇ Flat regions

◇ Other non-convex regions

Therefore:

> **NAG does not guarantee finding the global minimum.**

---

## 12. Advantages of NAG

◇ Uses accumulated momentum from previous updates.

◇ Looks ahead before calculating the gradient.

◇ Provides more informed updates.

◇ Can improve convergence compared with basic Gradient Descent.

◇ Can help control overshooting.

---

## 13. Limitations of NAG

◇ Depends on appropriate learning-rate and momentum settings.

◇ Oscillation can still occur with unsuitable hyperparameters.

◇ Does not guarantee finding the global minimum.

◇ More complex than basic Momentum because of the look-ahead step.

---

## 14. Real-Life Intuition

Imagine driving downhill. 🚗

### Momentum

You check the road **where you currently are** and decide how to move.

### NAG

You look **slightly ahead**, check what the road is doing, and then decide how to adjust your movement.

So:

> **Momentum reacts to where you are.**

> **NAG looks ahead to where momentum is taking you.**

---

# Final Summary

> **Nesterov Accelerated Gradient (NAG) is a momentum-based optimization algorithm that looks ahead in the direction of accumulated momentum, calculates the gradient at that look-ahead position, and then makes the actual weight update.**

### Remember:

**Momentum:**  
`Current Position → Gradient → Update`

**NAG:**  
`Current Position → Look Ahead → Gradient → Update`

> **NAG = Momentum + Look Ahead** 👀⚡