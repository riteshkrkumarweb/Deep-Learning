# Adam Optimizer

Adam stands for Adaptive Moment Estimation.

Adam combines the ideas of:
    1. Momentum
    2. RMSProp

## Why Adam?

Momentum:
    - Uses the moving average of past gradients.
    - Helps make the optimization direction smoother.
    - Reduces unnecessary oscillations.

RMSProp:
    - Uses the moving average of squared gradients.
    - Adapts the learning rate for each parameter.
    - Helps handle parameters with different gradient sizes.

Therefore:

    Adam = Momentum + RMSProp


## What Does Adam Maintain?

Adam maintains two moving averages:

    1. First Moment (m)
       - Moving average of gradients.
       - Similar to Momentum.

    2. Second Moment (v)
       - Moving average of squared gradients.
       - Similar to RMSProp.


## Bias Correction

At the beginning of training, m and v start from zero.

Because of this, their initial values can be biased toward zero.

Adam uses bias correction to compensate for this effect.

    Corrected m → m_hat
    Corrected v → v_hat


## Hyperparameters We Set Manually

Usually we set:

    Learning Rate (η)
        Default: 0.001

    Beta 1 (β1)
        Default: 0.9
        Controls the decay of the first moment.
        Related to Momentum.

    Beta 2 (β2)
        Default: 0.999
        Controls the decay of the second moment.
        Related to RMSProp.

    Epsilon (ε)
        Default: 1e-7 in Keras
        Prevents division by zero.



## Important Point

Adam does NOT mean:

    Momentum + Learning Rate Decay

Instead:

    Adam = Momentum + RMSProp

Learning-rate decay is a separate technique.

Adam can be used together with learning-rate schedules if required.


## Keras Example

from keras.optimizers import Adam

adam = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=adam,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


## Simple Intuition

Momentum:
    "Which direction should I move?"

RMSProp:
    "How large should my step be for each parameter?"

Adam:
    "Use both pieces of information to make a better update."


## Advantages of Adam

    - Usually converges faster than basic Gradient Descent.
    - Combines Momentum and RMSProp.
    - Adapts the learning rate for each parameter.
    - Works well for many deep learning problems.
    - Usually requires less manual learning-rate tuning.


## Important Limitation

Adam does not guarantee finding the global minimum.

Its performance depends on:
    - Learning rate
    - β1
    - β2
    - Model architecture
    - Dataset
    - Loss function


## One-Line Summary

Adam is an adaptive optimizer that combines Momentum's
gradient averaging with RMSProp's adaptive learning rates.