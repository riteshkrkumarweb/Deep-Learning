# AdaGrad

##  What is AdaGrad?

AdaGrad (Adaptive Gradient) is an optimization algorithm
that automatically adapts the learning rate for each parameter.

Main idea:

◇ Large accumulated gradients → smaller effective learning rate
◇ Small accumulated gradients → relatively larger effective learning rate


##  Why Do We Need AdaGrad?

In normal Gradient Descent, the learning rate is generally fixed.

Problem:

◇ Different parameters may need different step sizes.

AdaGrad solves this by automatically scaling the learning rate
for each parameter.


##  How AdaGrad Works

AdaGrad keeps a memory of the squared gradients.

At every training step:

◇ Calculate the current gradient
◇ Square the gradient
◇ Add it to the accumulated value
◇ Use the accumulated value to scale the learning rate
◇ Update the parameter


##  Why Does AdaGrad Square the Gradient?

Gradients can be positive or negative.

If we simply added them, positive and negative gradients
could cancel each other.

Squaring:

◇ Prevents positive and negative gradients from cancelling
◇ Measures the magnitude of the gradient

Example:

+2 → 4
-2 → 4


##  Accumulated Squared Gradients

AdaGrad maintains an accumulated value called v.

v represents the running total of squared gradients
from previous and current training steps.

Important:

◇ v can increase
◇ v can stay the same
◇ v does not decrease in standard AdaGrad

Therefore, the accumulated value generally becomes
larger as training continues.


##  Important Terms

w_t → Current weight before the update

w_t+1 → New weight after the update

t → Current training step / iteration

Gradient → Direction and magnitude of the loss change

v_t → Current accumulated squared gradients

v_t-1 → Previous accumulated squared gradients

η → Base learning rate

ε → Very small value used for numerical stability

Effective Learning Rate → Actual learning-rate factor used
                           after AdaGrad scaling


##  Base Learning Rate

The base learning rate is the value we initially set.

Example:

learning_rate = 0.01

Important:

◇ This is the base learning rate
◇ AdaGrad does not simply change this setting
◇ Instead, it calculates an effective learning rate
  for each parameter during the update


##  Effective Learning Rate

Effective learning rate means the actual learning-rate factor
used for a parameter after AdaGrad's scaling.

The effective learning rate depends on:

◇ Base learning rate
◇ Accumulated squared gradients

Therefore, different parameters can have different
effective learning rates.


##  How Does the Learning Rate Change?

Early training:

◇ Accumulated gradients are relatively small
◇ Effective learning rate is relatively larger

Later training:

◇ Accumulated gradients become larger
◇ Effective learning rate becomes smaller

So:

Large accumulated gradients → Smaller effective LR

Small accumulated gradients → Relatively larger effective LR


##  Why Does the Effective Learning Rate Decrease?

AdaGrad continuously adds squared gradients to its memory.

Because squared gradients are never negative:

◇ The accumulated value cannot decrease
◇ The denominator used for scaling generally becomes larger
◇ The effective learning rate decreases or stays the same


##  Simple Example

Suppose:

Base learning rate = 0.01

Gradients:

Step 1 → 2
Step 2 → 3
Step 3 → 1

The accumulated squared gradients become:

Step 1 → 4
Step 2 → 13
Step 3 → 14

Therefore, the effective learning rate becomes smaller
as the accumulated value increases.


##  AdaGrad and Sparse Data

AdaGrad is particularly useful for sparse datasets.

Sparse data means:

◇ Most values are zero
◇ Only a small number of values are non-zero

Examples:

◇ Text data
◇ Bag-of-Words
◇ TF-IDF
◇ One-hot encoded features
◇ Recommendation systems


##  Why Is AdaGrad Good for Sparse Features?

Suppose we have:

Feature A → Appears frequently
Feature B → Appears rarely

Frequent feature:

◇ Receives gradients more often
◇ Accumulates larger squared gradients
◇ Gets a smaller effective learning rate

Rare feature:

◇ Receives gradients less often
◇ Accumulates smaller squared gradients
◇ Gets a relatively larger effective learning rate

Therefore, AdaGrad is especially useful when some features
are frequent and others are rare.


##  Main Advantages of AdaGrad

◇ Automatically adapts the learning rate

◇ Gives different parameters different effective learning rates

◇ Particularly useful for sparse features

◇ Rare features can receive relatively larger updates

◇ Reduces the need to use one fixed learning rate
  for every parameter


##  Main Disadvantages of AdaGrad

◇ AdaGrad continuously accumulates squared gradients

◇ It does not forget older gradients

◇ The accumulated value can become very large

◇ The effective learning rate can become extremely small

◇ Updates can become very small

◇ Learning may become very slow or effectively stop


##  AdaGrad in Complex Neural Networks

AdaGrad does not guarantee convergence to the global minimum.

In complex, non-convex neural networks:

◇ The optimizer may stop making meaningful progress
  before reaching a good minimum

◇ This can happen because the effective learning rate
  becomes too small over time

Important:

Do NOT say:

"AdaGrad cannot converge to the minimum."

Better statement:

"AdaGrad may fail to reach a good minimum in complex
neural networks because its effective learning rate can
become too small over time."


##  AdaGrad vs SGD

SGD:

◇ Uses the current gradient
◇ Generally uses a fixed learning rate

AdaGrad:

◇ Uses the current gradient
◇ Remembers accumulated squared gradients
◇ Adapts the learning rate


One-line comparison:

SGD → Current gradient

AdaGrad → Accumulated squared gradients to adapt
           the learning rate


##  AdaGrad vs Momentum

Momentum:

◇ Remembers previous movement
◇ Uses velocity
◇ Helps reduce zig-zagging
◇ Helps maintain movement in consistent directions

AdaGrad:

◇ Remembers accumulated squared gradients
◇ Adapts the learning rate


One-line comparison:

Momentum → Past movement

AdaGrad → Accumulated squared gradients


##  AdaGrad vs NAG

SGD → Current gradient

Momentum → Past movement

NAG → Look-ahead + gradient

AdaGrad → Accumulated squared gradients
          → Adaptive learning rate


##  AdaGrad Flow

Current gradient
        ↓
Square the gradient
        ↓
Accumulate the squared gradient
        ↓
Calculate the effective learning rate
        ↓
Update the parameter
        ↓
Repeat



##  Why Was RMSProp Introduced?

AdaGrad's problem:

Old gradients are accumulated forever
        ↓
Accumulated value keeps increasing
        ↓
Effective learning rate keeps shrinking
        ↓
Learning can become too slow


RMSProp improves this idea by:

◇ Giving more importance to recent squared gradients
◇ Gradually forgetting older gradients
◇ Preventing the effective learning rate from shrinking
  too aggressively


##  Easy Memory Trick

AdaGrad = Adaptive Learning Rate

Remember:

Gradient
    ↓
Square
    ↓
Accumulate
    ↓
Scale Learning Rate
    ↓
Update Weight


Large accumulated gradient → Slow down

Small accumulated gradient → Relatively speed up


##  One-Line Definition

AdaGrad is an optimizer that adapts the learning rate
for each parameter using the accumulated squared gradients.


# Sparse Data

Sparse data = Data in which most values are 0
and only a few values are non-zero.

Example:
[0, 0, 1, 0, 0, 1, 0]

Examples:
◇ Text data
◇ One-hot encoded data
◇ Recommendation systems

AdaGrad works well with sparse data because
rare features can get relatively larger updates.