# ============================================================
# KERAS TUNER 🎯
# ============================================================

# Keras Tuner
# → Automatically searches for good hyperparameter combinations
#   for a Keras model.


# ============================================================
# 1. SEARCH SPACE 🔍
# ============================================================

# Search Space
# → All possible hyperparameter values/combinations
#   that the tuner is allowed to try.

# Example:

hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])
hp.Int("units", 32, 128, step=32)

# Possible combinations:
#
# Adam + 32
# Adam + 64
# Adam + 96
# Adam + 128
# SGD + 32
# SGD + 64
# ...
#
# Total = 3 × 4 = 12 combinations


# ============================================================
# 2. HYPERPARAMETER TYPES
# ============================================================

# hp.Int()
# → Used for integer values.

hp.Int("units", 32, 128, step=32)

# 32, 64, 96, 128


# hp.Float()
# → Used for decimal values.

hp.Float(
    "learning_rate",
    1e-4,
    1e-2,
    sampling="log"
)


# hp.Choice()
# → Selects one value from specific choices.

hp.Choice(
    "optimizer",
    ["adam", "sgd", "rmsprop"]
)


# hp.Boolean()
# → Selects True or False.

hp.Boolean("use_dropout")


# hp.Fixed()
# → Keeps a hyperparameter fixed.

hp.Fixed("learning_rate", 0.001)


# ============================================================
# 3. IMPORTANT CONCEPTS
# ============================================================

# Trial 🧪
# → One complete hyperparameter configuration tested.

# Example:
# Trial 1 → Adam + 64 neurons
# Trial 2 → SGD + 128 neurons
# Trial 3 → RMSprop + 32 neurons


# Objective 🎯
# → Metric that the tuner tries to improve.

objective="val_accuracy"

# val_accuracy → maximize
# val_loss     → minimize


# Hypermodel
# → Function that builds and returns the model.

def build_model(hp):
    ...
    return model


# ============================================================
# 4. RANDOM SEARCH 🎲
# ============================================================

# What is Random Search?
#
# → Randomly selects hyperparameter combinations
#   from the search space and tests them.
#
# Example:
#
# Search Space:
# 12 possible combinations
#
# max_trials=5
#
# → Randomly tests 5 combinations
# → Compares their results
# → Selects the best one 🏆


# Code:

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    directory="tuner_results",
    project_name="random_search"
)


# Important Parameters:
#
# hypermodel
# → Model-building function.
#
# objective
# → Metric to optimize.
#
# max_trials
# → Maximum number of configurations to test.
#
# executions_per_trial
# → Number of times the same configuration is trained.
#
# seed
# → Controls randomness.
#
# directory
# → Saves tuning results.
#
# project_name
# → Name of the tuning project.
#
# overwrite
# → Whether to overwrite previous results.


# ============================================================
# 5. BAYESIAN OPTIMIZATION 🧠
# ============================================================

# What is Bayesian Optimization?
#
# → Uses results from previous trials to guide
#   the next hyperparameter combinations.
#
# Instead of randomly searching every time:
#
# Previous results
#       ↓
# Learn which areas look promising
#       ↓
# Choose the next configuration
#       ↓
# Get result
#       ↓
# Use the new result again


# Example:

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


# Important Parameters:
#
# hypermodel
# → Model-building function.
#
# objective
# → Metric to optimize.
#
# max_trials
# → Maximum number of trials.
#
# num_initial_points
# → Initial trials used to collect information.
#
# alpha
# → Controls the assumed noise in the Gaussian Process.
#
# beta
# → Controls how much the tuner explores new/uncertain areas.
#
# Low beta
# → Focuses more on promising combinations.
#
# High beta
# → Tries more new/uncertain combinations.
#
# seed
# → Controls randomness.
#
# directory
# → Saves tuning results.
#
# project_name
# → Name of the tuning project.


# ============================================================
# 6. HYPERBAND ⚡
# ============================================================

# What is Hyperband?
#
# → Tests many hyperparameter configurations with
#   limited training first.
#
# → Weak configurations receive less training.
#
# → Promising configurations receive more training.
#
# → Saves training time and resources.


# Example:

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


# Important Parameters:
#
# hypermodel
# → Model-building function.
#
# objective
# → Metric to optimize.
#
# max_epochs
# → Maximum epochs a configuration can receive.
#
# factor
# → Controls how aggressively configurations are reduced.
#
# Example:
# 9 configurations → roughly 3 continue
# 3 configurations → roughly 1 continues
#
# hyperband_iterations
# → Number of times the Hyperband search process is repeated.
#
# seed
# → Controls randomness.
#
# directory
# → Saves tuning results.
#
# project_name
# → Name of the tuning project.


# ============================================================
# 7. COMMON PARAMETERS
# ============================================================

# hypermodel
# → Builds the model.

# objective
# → Metric to optimize.

# seed
# → Controls randomness.

# directory
# → Folder for tuner results.

# project_name
# → Name of the tuning project.

# overwrite
# → Overwrite existing tuning results.

# executions_per_trial
# → Train the same configuration multiple times.

# max_retries_per_trial
# → Retry a failed trial.

# max_consecutive_failed_trials
# → Stop after too many consecutive failed trials.


# ============================================================
# 8. ADVANCED PARAMETERS
# ============================================================

# max_model_size
# → Maximum allowed model size/number of parameters.

# distribution_strategy
# → Distributes training across multiple devices.

# tuner_id
# → Identifier for advanced/distributed tuning.

# tune_new_entries
# → Controls whether new hyperparameters are tuned.

# allow_new_entries
# → Controls whether new hyperparameters can be added.


# ============================================================
# 9. START SEARCH 🚀
# ============================================================

tuner.search(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)

# search()
# → Starts the tuning process.


# ============================================================
# 10. GET BEST HYPERPARAMETERS 🏆
# ============================================================

best_hp = tuner.get_best_hyperparameters(
    num_trials=1
)[0]

# → Returns the best hyperparameter configuration.


# ============================================================
# 11. GET BEST MODEL
# ============================================================

best_model = tuner.get_best_models(
    num_models=1
)[0]

# → Returns the best trained model.


# ============================================================
# 12. VIEW RESULTS
# ============================================================

tuner.results_summary()

# → Shows tuning results.


# ============================================================
# 13. VIEW SEARCH SPACE
# ============================================================

tuner.search_space_summary()

# → Shows the hyperparameters being tuned.


# ============================================================
# 14. QUICK COMPARISON 🧠
# ============================================================

# Random Search 🎲
#
# → Randomly selects combinations.
# → Uses max_trials.
#
# Example:
# 20 possible combinations
# max_trials=5
# → Tests up to 5 combinations.


# Bayesian Optimization 🧠
#
# → Uses previous results to guide the next search.
# → Uses max_trials.
#
# Example:
# First 5 trials → collect information
# Later trials → use previous results to guide the search.


# Hyperband ⚡
#
# → Tests many configurations with limited training.
# → Gives more training to promising configurations.
# → Uses max_epochs and factor.
#
# Example:
# Many configurations
#       ↓
# Less training
#       ↓
# Weak configurations reduced
#       ↓
# Promising configurations
#       ↓
# More training


# ============================================================
# 15. MOST IMPORTANT PARAMETERS ⭐
# ============================================================

# Random Search:
#
# max_trials
# → Maximum number of trials.


# Bayesian Optimization:
#
# max_trials
# → Maximum number of trials.
#
# num_initial_points
# → Initial trials used to collect information.
#
# alpha
# → Assumed noise level.
#
# beta
# → Amount of exploration.


# Hyperband:
#
# max_epochs
# → Maximum training epochs.
#
# factor
# → Controls reduction of configurations.
#
# hyperband_iterations
# → Number of Hyperband search repetitions.


# ============================================================
# 16. BASIC WORKFLOW 🔄
# ============================================================

# 1. Define hyperparameter search space
#
# 2. Create build_model(hp)
#
# 3. Choose a tuner
#
# 4. Set objective
#
# 5. Run tuner.search()
#
# 6. Get best hyperparameters
#
# 7. Build/retrain final model
#
# 8. Evaluate final model

# For more u can visit the link or documentation of the keras_tuner https://keras.io/keras_tuner/