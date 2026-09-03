# RMSProp Optimizer

## What is RMSProp?

RMSProp stands for **Root Mean Square Propagation**.

RMSProp is an adaptive learning-rate optimizer used in Machine Learning and Deep Learning.

It is considered an improvement over AdaGrad because it addresses AdaGrad's main problem of continuously decreasing the learning rate.

---

## Why RMSProp?

### Problem with AdaGrad

AdaGrad continuously accumulates squared gradients.

As training continues:

Accumulated squared gradients increase
        ↓
Effective learning rate decreases
        ↓
Updates become smaller and smaller
        ↓
Learning can become extremely slow

### RMSProp Solution

RMSProp uses an **exponentially weighted moving average** of squared gradients.

This means:

Recent gradients → More influence
Older gradients → Less influence
Very old gradients → Very little influence

So the learning rate does not decrease as aggressively as AdaGrad.

---

## What Does Beta (β) Do?

β is the **decay factor**.

It controls how much past gradient information is remembered.

For example:

β = 0.9

Approximately:

90% → Previous moving average
10% → Current squared gradient

Higher β:
→ Remembers previous information more
→ Older information decays more slowly

Lower β:
→ Gives more importance to recent gradients
→ Older information decays faster

Simple definition:

**β controls how quickly old gradient information is forgotten.**

---

## AdaGrad vs RMSProp

AdaGrad:
→ Accumulates all past squared gradients
→ Does not forget old information
→ Accumulated value keeps increasing
→ Effective learning rate can become too small

RMSProp:
→ Uses a moving average of squared gradients
→ Gradually forgets old information
→ Recent gradients have more influence
→ Prevents learning rate from shrinking too aggressively

---

## Sparse Data

RMSProp does NOT only work with sparse data.

RMSProp can work with:

→ Sparse data
→ Dense data
→ Neural networks
→ Non-convex optimization problems

AdaGrad is particularly useful for sparse data.

---

## RMSProp and Neural Networks

Neural networks usually involve non-convex optimization.

Their loss landscape can contain:

→ Local minima
→ Saddle points
→ Flat regions
→ Steep regions

RMSProp helps improve the optimization process by adapting the learning rate for different parameters.

However:

**RMSProp does NOT guarantee finding the global minimum.**

---

## Effective Learning Rate

RMSProp adjusts the learning rate for each parameter based on the recent history of its gradients.

Large recent gradients:
→ Smaller effective learning rate

Small recent gradients:
→ Larger effective learning rate

This allows different parameters to have different effective learning rates.

---

## Advantages of RMSProp

→ Adaptive learning rate

→ Gives different parameters different effective learning rates

→ Reduces AdaGrad's aggressive learning-rate decay

→ Gives more importance to recent gradients

→ Works well for many neural-network optimization problems

→ Can work with both sparse and dense data

---

## Disadvantages of RMSProp

→ Does not guarantee the global minimum

→ Performance depends on hyperparameters such as learning rate and β

→ Does not solve every problem of non-convex optimization

---

## Evolution of Optimizers

SGD
 ↓
Momentum
 ↓
AdaGrad
 ↓
RMSProp
 ↓
Adam

SGD:
→ Basic gradient-based optimization

Momentum:
→ Uses previous movement

AdaGrad:
→ Adapts learning rate using accumulated squared gradients

RMSProp:
→ Uses a decaying average of squared gradients

Adam:
→ Combines ideas from Momentum and RMSProp

---

## One-Line Definition

RMSProp is an adaptive optimizer and an improvement over AdaGrad that uses an exponentially decaying average of squared gradients to prevent the effective learning rate from becoming too small.