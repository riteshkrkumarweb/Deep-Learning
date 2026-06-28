# Deep Learning Basics Summary

## 1. Weight

**Definition:** A weight determines how important an input feature is to
the prediction.

**Key Points** - Positive weight → increases output. - Negative weight →
decreases output. - Larger absolute value → stronger influence.

**Example** If Study Hours has a weight of **10**, every extra hour
contributes **10** units to the prediction.

------------------------------------------------------------------------

## 2. Bias

**Definition:** A bias is a constant value added to the weighted sum. It
shifts the prediction independently of the inputs.

**Example** Taxi Fare = Base Fare + (Distance × Price per km)

-   Base Fare = Bias

------------------------------------------------------------------------

## 3. Summation (Σ)

**Definition:** Summation is the process of multiplying each input by
its weight, adding the results together, and then adding the bias.

**Formula**

z = x₁w₁ + x₂w₂ + ... + b

**Example**

Input = 4, 6

Weights = 2, 3

Bias = 5

z = (4×2) + (6×3) + 5 = 31

------------------------------------------------------------------------

## 4. Activation Function

**Definition:** An activation function takes the summation value and
produces the neuron's final output.

**Purpose** Without an activation function, a neural network behaves
like a linear model.

**Example** If z = 31:

-   ReLU → 31
-   Sigmoid → Value between 0 and 1
-   Step Function → 1

------------------------------------------------------------------------

## 5. Step Function

**Definition:** The simplest activation function. It outputs only 0 or
1.

**Rule**

If z ≥ 0 → 1

Else → 0

**Example**

z = 15 → Output = 1

z = -4 → Output = 0

------------------------------------------------------------------------

## 6. Perceptron

**Definition:** A perceptron is the simplest artificial neuron. It takes
inputs, applies weights and bias, computes a weighted sum, applies an
activation function, and produces an output.

**Formula**

y = f(x₁w₁ + x₂w₂ + ... + b)

**Flow**

Input → Weight → Bias → Summation → Activation Function → Output

------------------------------------------------------------------------

## 7. Geometrical Intuition of Perceptron

**Definition:** A perceptron is a linear classifier that separates data
using a decision boundary.

**Decision Boundary** - 2D → Line - 3D → Plane - Higher Dimensions →
Hyperplane

**Weights** Rotate the decision boundary.

**Bias** Shifts the decision boundary.

**Example**

Blue Points

● ●

------------------ (Decision Boundary)

○ ○ ○

Red Points

------------------------------------------------------------------------

