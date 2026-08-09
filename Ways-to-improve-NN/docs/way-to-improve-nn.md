# Ways to Improve NN performace

## A. Fine-Tuning Hyperparameters

**Definition:**  
Fine-tuning means adjusting the neural network's hyperparameters to improve its performance.

**Main hyperparameters:**
* Number of hidden layers
* Number of neurons per layer
* Learning rate
* Learning rate scheduler
* Optimizer
* Batch size
* Activation function
* Epochs


## 1. Number of Hidden Layers

**Definition:**  
Determines the depth and complexity of the neural network.

**Example:**  
1 hidden layer → simpler model  
3 hidden layers → can learn more complex patterns.


## 2. Number of Neurons

**Definition:**  
Determines the capacity of each hidden layer.

**Example:**  
1 hidden layer with 128 neurons → 128 neurons in that layer.

More neurons → more model capacity, but may increase overfitting.


## 3. Learning Rate

**Definition:**  
Controls the size of weight updates during training.

**Example:**  
Learning rate = 0.01 → relatively larger updates  
Learning rate = 0.0001 → smaller updates

Too large → unstable training  
Too small → slow training


## 3a. Learning Rate Scheduler

**Definition:**  
Automatically changes the learning rate during training.

**Example:**

0.01 → 0.001 → 0.0001

Large/faster steps initially → smaller/finer steps later.


## 4. Optimizer

**Definition:**  
An algorithm that updates the weights and biases to minimize the loss.

**Examples:**
* SGD
* Adam
* RMSprop


## 5. Batch Size

**Definition:**  
The number of training samples used to calculate the gradient before one weight update.

### Batch Gradient Descent

Batch size = entire dataset

Example:

320 samples → 1 update/epoch


### Stochastic Gradient Descent

Batch size = 1

Example:

320 samples → 320 updates/epoch


### Mini-Batch Gradient Descent

Batch size = small group

Example:

320 samples, batch size = 32

320 ÷ 32 = 10 updates/epoch

Common starting values → 32 or 64


## 6. Activation Function

**Definition:**  
An activation function introduces non-linearity, allowing the neural network to learn complex patterns.

**Examples:**
* ReLU → commonly used in hidden layers
* Sigmoid → binary classification output
* Softmax → multi-class classification output
* Tanh → sometimes used in hidden layers


## 7. Epochs

**Definition:**  
One complete pass through the entire training dataset.

**Example:**

epochs = 100

→ Maximum of 100 passes through the training data.


## Early Stopping

**Definition:**  
Early stopping automatically stops training when validation performance stops improving.

**Example:**

Maximum epochs = 1000

Validation loss improves until epoch 120.

→ Training may stop around epoch 125 instead of continuing to 1000.

**Purpose:**
* Reduce overfitting
* Save training time
* Avoid unnecessary training


## Keras Callback

**Definition:**  
A callback allows us to perform an action during training.

**Example:**

EarlyStopping is a Keras callback.

EarlyStopping(
    monitor='val_loss',
    patience=5
)

patience = 5 → wait 5 epochs for improvement before stopping.

# B. Problem That we Face in the NN and with the solution 

##  Vanishing Gradient

**Definition:**  
The gradients (partial derivatives of the loss function) become extremely small during backpropagation.

→ Earlier layers learn very slowly.

**Solutions:**
* ReLU
* Better weight initialization
* Batch normalization
* Suitable network architecture


## Exploding Gradient

**Definition:**  
The gradients (partial derivatives of the loss function) become extremely large.

→ Training becomes unstable.

**Solutions:**
* Gradient clipping
* Proper weight initialization
* Suitable learning rate
* Batch normalization


## Not Enough Data

**Definition:**  
A neural network may not learn general patterns well when there is insufficient training data.

**Solutions:**
* Collect more data
* Data augmentation
* Transfer learning


## Slow Training

**Definition:**  
Training takes too long to reach good performance.

**Possible solutions:**
* Suitable batch size
* Suitable learning rate
* Learning-rate scheduler
* Better optimizer
* GPU
* Appropriate model architecture


## Overfitting

**Definition:**  
Overfitting occurs when the model learns the training data too closely and performs poorly on unseen data.

**Example:**

Training accuracy = 99%  
Validation accuracy = 75%

**Solutions:**
* More data
* Regularization
* Dropout
* Reduce model complexity
* Early stopping
* Data augmentation


## Transfer Learning

**Definition:**  
Transfer learning means reusing knowledge learned by a pre-trained model for a new, related task.

**Example:**

Pre-trained model trained on millions of images

→ Reuse its learned features

→ Train it for cat-vs-dog classification.

**Benefits:**
* Less training data may be required
* Faster training
* Can improve performance on small datasets


# ⭐ Final Revision

Batch size → How many samples are used for one weight update.

Learning rate → Size of the weight update.

Learning rate scheduler → Changes the learning rate during training.

Epoch → One complete pass through the dataset.

Early stopping → Stops training when validation performance stops improving.

Optimizer → Updates weights and biases to minimize loss.

Activation function → Adds non-linearity to the network.

Transfer learning → Reuses knowledge from a pre-trained model.

Fine-tuning → Adjusts hyperparameters/model parameters to improve performance.