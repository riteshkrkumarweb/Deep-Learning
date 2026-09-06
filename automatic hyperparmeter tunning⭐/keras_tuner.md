# KERAS TUNER 🎯

> Automatically searches for good hyperparameter combinations for a Keras model.

---

# 1. SEARCH SPACE 🔍

## What is Search Space?

The **search space** is the collection of all possible hyperparameter values and combinations that the tuner is allowed to try.

### Example

```python
hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])

hp.Int("units", 32, 128, step=32)
```

Possible values:

```text
Optimizer:
adam
sgd
rmsprop

Units:
32
64
96
128
```

Possible combinations:

```text
Adam + 32
Adam + 64
Adam + 96
Adam + 128

SGD + 32
SGD + 64
SGD + 96
SGD + 128

RMSprop + 32
RMSprop + 64
RMSprop + 96
RMSprop + 128
```

Total:

```text
3 × 4 = 12 combinations
```

---

# 2. HYPERPARAMETER TYPES ⚙️

## `hp.Int()`

Used when the hyperparameter is an **integer**.

```python
hp.Int("units", 32, 128, step=32)
```

Possible values:

```text
32, 64, 96, 128
```

---

## `hp.Float()`

Used when the hyperparameter is a **decimal/continuous value**.

```python
hp.Float(
    "learning_rate",
    1e-4,
    1e-2,
    sampling="log"
)
```

Example range:

```text
0.0001 → 0.01
```

`sampling="log"` means the values are sampled on a logarithmic scale rather than uniformly.

---

## `hp.Choice()`

Selects one value from a predefined list.

```python
hp.Choice(
    "optimizer",
    ["adam", "sgd", "rmsprop"]
)
```

Possible values:

```text
adam
sgd
rmsprop
```

---

## `hp.Boolean()`

Selects between `True` and `False`.

```python
hp.Boolean("use_dropout")
```

Possible values:

```text
True
False
```

Example:

```python
if hp.Boolean("use_dropout"):
    model.add(Dropout(0.5))
```

---

## `hp.Fixed()`

Keeps a hyperparameter fixed instead of tuning it.

```python
hp.Fixed("learning_rate", 0.001)
```

The tuner will always use:

```text
learning_rate = 0.001
```

---

# 3. IMPORTANT CONCEPTS 🧠

## Trial 🧪

A **trial** is one complete hyperparameter configuration that is tested.

Example:

```text
Trial 1 → Adam + 64 neurons
Trial 2 → SGD + 128 neurons
Trial 3 → RMSprop + 32 neurons
```

Each trial trains/evaluates a model using its selected hyperparameters.

---

## Objective 🎯

The **objective** is the metric that the tuner tries to improve.

Example:

```python
objective="val_accuracy"
```

For `val_accuracy`:

```text
Higher = Better
→ maximize
```

For `val_loss`:

```text
Lower = Better
→ minimize
```

---

## Hypermodel 🏗️

A **hypermodel** is the function that receives `hp` and builds/returns the model.

```python
def build_model(hp):

    ...

    return model
```

The tuner calls this function with different hyperparameter values to create different model configurations.

---

# 4. RANDOM SEARCH 🎲

## What is Random Search?

Random Search randomly selects hyperparameter combinations from the search space and tests them.

Example:

```text
Search Space
     ↓
12 possible combinations
     ↓
max_trials = 5
     ↓
Randomly select up to 5 combinations
     ↓
Train and evaluate
     ↓
Compare results
     ↓
Best configuration 🏆
```

### Example

```python
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    directory="tuner_results",
    project_name="random_search"
)
```

## Important Parameters

### `hypermodel`

The model-building function.

```python
hypermodel=build_model
```

---

### `objective`

The metric that should be optimized.

```python
objective="val_accuracy"
```

---

### `max_trials`

Maximum number of different hyperparameter configurations to test.

```python
max_trials=10
```

If there are 100 possible combinations and:

```python
max_trials=10
```

the tuner will test at most 10 configurations.

---

### `executions_per_trial`

Number of times the **same configuration** is trained.

```python
executions_per_trial=3
```

Useful when training results vary because of randomness.

Conceptually:

```text
Same configuration
       ↓
Train 1 → accuracy = 91%
Train 2 → accuracy = 90%
Train 3 → accuracy = 92%
       ↓
Average result
```

---

### `seed`

Controls randomness.

```python
seed=42
```

Using the same seed can make the search more reproducible.

---

### `directory`

Folder where tuner results are stored.

```python
directory="tuner_results"
```

---

### `project_name`

Name of the tuning project.

```python
project_name="random_search"
```

---

### `overwrite`

Controls whether previous tuning results are overwritten.

```python
overwrite=True
```

---

# 5. BAYESIAN OPTIMIZATION 🧠

## What is Bayesian Optimization?

Bayesian Optimization uses the results from previous trials to guide the selection of future hyperparameter combinations.

Unlike Random Search, it does not treat every future trial as completely independent.

### Basic idea

```text
Previous trials
      ↓
Learn which regions look promising
      ↓
Choose next configuration
      ↓
Train model
      ↓
Get result
      ↓
Use result to guide next trial
      ↓
Repeat
```

### Example

```python
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=20,
    num_initial_points=5,
    alpha=0.0001,
    beta=2.6,
    seed=42,
    directory="tuner_results",
    project_name="bayesian_search"
)
```

## Important Parameters

### `hypermodel`

Model-building function.

```python
hypermodel=build_model
```

### `objective`

Metric to optimize.

```python
objective="val_accuracy"
```

### `max_trials`

Maximum number of trials.

```python
max_trials=20
```

### `num_initial_points`

Number of initial trials used to collect information before Bayesian guidance becomes useful.

```python
num_initial_points=5
```

Conceptually:

```text
First 5 trials
      ↓
Collect information
      ↓
Use information to guide later trials
```

### `alpha`

Controls the assumed noise in the Gaussian Process used by the Bayesian optimizer.

```python
alpha=0.0001
```

### `beta`

Controls the balance between **exploration** and **exploitation**.

```python
beta=2.6
```

Lower beta:

```text
More focus on promising areas
→ Exploitation
```

Higher beta:

```text
More exploration of uncertain/new areas
→ Exploration
```

### `seed`

Controls randomness.

```python
seed=42
```

### `directory`

Stores tuning results.

```python
directory="tuner_results"
```

### `project_name`

Name of the tuning project.

```python
project_name="bayesian_search"
```

---

# 6. HYPERBAND ⚡

## What is Hyperband?

Hyperband tests many hyperparameter configurations with limited training first.

Weak configurations receive less training, while promising configurations receive more training.

### Basic idea

```text
Many configurations
       ↓
Small amount of training
       ↓
Evaluate performance
       ↓
Remove weak configurations
       ↓
Give more training to promising configurations
       ↓
Repeat
       ↓
Best configuration 🏆
```

This can save training time and computational resources.

---

## Example

```python
tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    hyperband_iterations=2,
    seed=42,
    directory="tuner_results",
    project_name="hyperband_search"
)
```

## Important Parameters

### `hypermodel`

Model-building function.

```python
hypermodel=build_model
```

### `objective`

Metric to optimize.

```python
objective="val_accuracy"
```

### `max_epochs`

Maximum number of epochs a configuration can receive.

```python
max_epochs=20
```

### `factor`

Controls how aggressively configurations are reduced.

```python
factor=3
```

Conceptual example:

```text
9 configurations
      ↓
roughly 3 continue

3 configurations
      ↓
roughly 1 continues
```

The exact scheduling depends on Hyperband's brackets/rungs, so this is a simplified illustration.

### `hyperband_iterations`

Number of times the Hyperband search process is repeated.

```python
hyperband_iterations=2
```

Higher values can provide more search but require more computation.

### `seed`

Controls randomness.

```python
seed=42
```

### `directory`

Stores tuning results.

```python
directory="tuner_results"
```

### `project_name`

Name of the tuning project.

```python
project_name="hyperband_search"
```

---

# 7. COMMON PARAMETERS 🔧

These parameters can be relevant across Keras Tuner algorithms.

### `hypermodel`

Builds the model.

```python
hypermodel=build_model
```

### `objective`

Metric to optimize.

```python
objective="val_accuracy"
```

### `seed`

Controls randomness.

```python
seed=42
```

### `directory`

Folder used to store tuner results.

```python
directory="tuner_results"
```

### `project_name`

Name of the tuning project.

```python
project_name="my_search"
```

### `overwrite`

Whether to overwrite existing tuning results.

```python
overwrite=True
```

### `executions_per_trial`

Number of times each trial is executed.

```python
executions_per_trial=3
```

### `max_retries_per_trial`

Number of times to retry a failed trial.

```python
max_retries_per_trial=2
```

### `max_consecutive_failed_trials`

Maximum number of consecutive failed trials before the search is stopped.

```python
max_consecutive_failed_trials=3
```

---

# 8. ADVANCED PARAMETERS ⚙️

### `max_model_size`

Maximum allowed model size/number of parameters.

This can prevent the tuner from selecting models that are too large.

---

### `distribution_strategy`

Used to distribute training across multiple devices.

---

### `tuner_id`

Identifier used in advanced/distributed tuning setups.

---

### `tune_new_entries`

Controls whether newly encountered hyperparameters should be tuned.

---

### `allow_new_entries`

Controls whether new hyperparameters can be added to the search space.

---

# 9. START THE SEARCH 🚀

After creating the tuner, use:

```python
tuner.search(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)
```

### `tuner.search()`

Starts the hyperparameter tuning process.

Conceptually:

```text
Search Space
      ↓
Tuner
      ↓
Trial 1
      ↓
Trial 2
      ↓
Trial 3
      ↓
...
      ↓
Best configuration
```

---

# 10. GET BEST HYPERPARAMETERS 🏆

```python
best_hp = tuner.get_best_hyperparameters(
    num_trials=1
)[0]
```

This returns the best hyperparameter configuration found by the tuner.

You can inspect individual values:

```python
print(best_hp.get("optimizer"))
print(best_hp.get("units"))
print(best_hp.get("learning_rate"))
```

---

# 11. GET BEST MODEL 🏆

```python
best_model = tuner.get_best_models(
    num_models=1
)[0]
```

This returns the best trained model found by the tuner.

---

# 12. VIEW RESULTS 📊

```python
tuner.results_summary()
```

Shows a summary of the tuning results.

Useful for seeing which trials performed best.

---

# 13. VIEW SEARCH SPACE 🔍

```python
tuner.search_space_summary()
```

Shows the hyperparameters that are included in the tuning search space.

---

# 14. QUICK COMPARISON 🧠

## Random Search 🎲

```text
Randomly selects hyperparameter combinations
```

Main parameter:

```python
max_trials
```

Example:

```text
20 possible combinations
max_trials = 5

→ Tests up to 5 combinations
```

Best when you want a simple search strategy and have a reasonable trial budget.

---

## Bayesian Optimization 🧠

```text
Uses previous results to guide future trials
```

Main parameters:

```python
max_trials
num_initial_points
alpha
beta
```

Example:

```text
First trials
     ↓
Collect information
     ↓
Later trials
     ↓
Use previous results to guide search
```

Best when each model training run is expensive and you want the search to learn from previous trials.

---

## Hyperband ⚡

```text
Tests many configurations with limited training
```

Main parameters:

```python
max_epochs
factor
hyperband_iterations
```

Example:

```text
Many configurations
       ↓
Less training
       ↓
Weak configurations removed
       ↓
Promising configurations
       ↓
More training
```

Best when you want to eliminate poor configurations early and save computation.

---

# 15. MOST IMPORTANT PARAMETERS ⭐

## Random Search 🎲

```python
max_trials
```

→ Maximum number of trials/configurations to test.

---

## Bayesian Optimization 🧠

```python
max_trials
```

→ Maximum number of trials.

```python
num_initial_points
```

→ Initial trials used to collect information.

```python
alpha
```

→ Assumed noise level in the Gaussian Process.

```python
beta
```

→ Controls exploration versus exploitation.

---

## Hyperband ⚡

```python
max_epochs
```

→ Maximum training epochs available to a configuration.

```python
factor
```

→ Controls how aggressively configurations are reduced.

```python
hyperband_iterations
```

→ Number of Hyperband search repetitions.

---

# 16. BASIC KERAS TUNER WORKFLOW 🔄

```text
1. Define the hyperparameter search space
              ↓
2. Create build_model(hp)
              ↓
3. Choose a tuner
              ↓
4. Set the objective
              ↓
5. Run tuner.search()
              ↓
6. Get the best hyperparameters
              ↓
7. Build/retrain the final model
              ↓
8. Evaluate the final model
```

### Simple structure

```python
import keras_tuner as kt

def build_model(hp):

    model = Sequential()

    units = hp.Int(
        "units",
        min_value=32,
        max_value=128,
        step=32
    )

    model.add(
        Dense(
            units,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    directory="tuner_results",
    project_name="my_search"
)


tuner.search(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)


best_hp = tuner.get_best_hyperparameters(
    num_trials=1
)[0]

best_model = tuner.get_best_models(
    num_models=1
)[0]
```

---

# 17. THE MAIN IDEA TO REMEMBER 🧠

```text
Hyperparameters
       ↓
Search Space
       ↓
Tuner
       ↓
Trials
       ↓
Model Training
       ↓
Objective Metric
       ↓
Compare Results
       ↓
Best Hyperparameters
       ↓
Best Model
```

### The three main tuning algorithms

```text
Random Search
→ Randomly chooses configurations

Bayesian Optimization
→ Learns from previous trials

Hyperband
→ Gives limited training first and
  gives more training to promising configurations
```

---

# 18. QUICK MEMORY TRICK 🧠

```text
Random Search
→ RANDOMLY SEARCH 🎲

Bayesian Optimization
→ LEARN FROM PREVIOUS RESULTS 🧠

Hyperband
→ STOP BAD MODELS EARLY ⚡
```

---

# 19. IMPORTANT DISTINCTION ⭐

Do not confuse these terms:

```text
Hyperparameter
    ↓
A setting you choose before/during model training
    ↓
Example:
learning_rate
units
dropout
optimizer
batch_size
```

```text
Trial
    ↓
One particular combination of hyperparameters
    ↓
Example:
optimizer = Adam
units = 64
dropout = 0.3
learning_rate = 0.001
```

```text
Objective
    ↓
What you use to decide which trial is better
    ↓
Example:
val_accuracy
val_loss
```

```text
Tuner
    ↓
The algorithm that decides which
hyperparameter configurations to test
    ↓
RandomSearch
BayesianOptimization
Hyperband
```

---

# 20. ONE-LINE SUMMARY 🚀

> **Keras Tuner automatically searches different hyperparameter configurations, trains models for those configurations, compares them using an objective metric, and helps you find a good-performing configuration.**

---

# 📚 DOCUMENTATION

Keras Tuner documentation:

https://keras.io/keras_tuner/
