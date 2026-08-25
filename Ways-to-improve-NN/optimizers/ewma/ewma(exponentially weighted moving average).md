# Exponentially Weighted Moving Average (EWMA)

## What is EWMA?

EWMA (Exponentially Weighted Moving Average) is a technique used to **smooth noisy or fluctuating data** and show the overall trend more clearly.

## Why do we use EWMA?

* Reduces the effect of sudden changes or noise
* Makes the data trend easier to understand
* Gives more importance to recent data
* Gradually reduces the influence of older data

## Recent Data Has Greater Weight

A newer data point has **more influence** than an older data point.

**Example:** Today's temperature has more influence on the smoothed temperature than yesterday's temperature.

## Older Data Loses Weight Over Time

As data becomes older, its influence **gradually decreases**.

The old data is not immediately removed; its influence becomes smaller and smaller as new data arrives.

**Example:**

```text
Day 1 → High influence
Day 2 → Less influence
Day 3 → Even less influence
Day 4 → Smaller influence
Day 5 → Very small influence
```

## What is a Smoothed Value?

A smoothed value changes **more gradually** than the original data because it considers previous information along with new data.

**Example:**

```text
Actual data:
30 → 20 → 40 → 25 → 35

Smoothed data:
9 → 12.3 → 20.61 → 21.927 → 25.8489
```

The actual data changes sharply, while the smoothed values change more gradually.

## Effect of Beta

Beta controls **how much EWMA remembers the past**.

### Beta = 0.98

* Very stable
* Very smooth
* Reacts very slowly to new changes
* Strong memory of the past

### Beta = 0.9

* Smooth
* Good balance between stability and responsiveness
* Commonly used in EWMA examples
* Not always the best value for every problem

### Beta = 0.1

* Very responsive
* Reacts quickly to new data
* Less smooth
* More affected by noise

### Easy Rule

**Higher beta → more stable, smoother, slower**

**Lower beta → more responsive, less smooth, faster**

## Does EWMA Always Move Upward?

No. EWMA can move **upward or downward**.

It does not force the data to have an upward trend.

If the current data is higher than the previous smoothed value, EWMA tends to move upward.

If the current data is lower than the previous smoothed value, EWMA tends to move downward.

## Important Difference

Do not think that EWMA simply follows the direction of the actual data from one day to the next.

EWMA uses:

**Previous smoothed value + current actual value**

For example:

```text
Actual:
30 → 20 ↓

EWMA:
9 → 12.3 ↑
```

This is possible because the current actual value (20) is still much higher than the previous EWMA value (9).

## EWMA in Deep Learning

EWMA is **not an optimizer itself**.

It is a technique used inside optimizers such as **Momentum and Adam** to maintain a moving average of gradients.

This helps reduce noisy or rapidly changing gradient updates.

## Quick Revision

* **EWMA** → Smooths changing/noisy data
* **Recent data** → More influence
* **Older data** → Less influence
* **Smoothed value** → Changes more gradually
* **Higher beta** → More stable and smoother
* **Lower beta** → More responsive
* **EWMA can move** → Both upward and downward
* **EWMA is used in** → Momentum and Adam
* **Main purpose** → Reduce fluctuations and capture the overall trend

## One-Line Definition

> **EWMA is a smoothing technique that gives more importance to recent data while gradually reducing the influence of older data.**


