# Gradient Descent — My Notes

## 1. Optimization

Optimization = finding parameters that minimize the loss.

Goal:
Loss ↓ → Better parameters


## 2. Gradient

Gradient = collection of partial derivatives of the loss with respect to the model parameters.

Gradient tells us how the loss changes with w and b.

Gradient Descent moves in the opposite direction of the gradient.


## 3. Gradient Descent

Gradient Descent repeatedly:

Prediction
↓
Loss
↓
Gradient
↓
Update w and b
↓
Repeat


## 4. Learning Rate

Learning rate controls the size of the update.

Small learning rate → small steps
Large learning rate → large steps


## 5. Epoch

1 Epoch = the entire dataset has been processed once.

Important:
Epoch does NOT mean all samples are processed at the same time.

Example:

BGD:
10 samples → processed together → 1 epoch

SGD:
1 → 2 → 3 → ... → 10 → 1 epoch


## 6. Batch

Batch = a group of samples processed before one parameter update.


## 7. Batch Size

Batch size = number of rows/samples used for one gradient calculation and one parameter update.

Example:

100 rows
Batch size = 20

20 rows → Update
20 rows → Update
20 rows → Update
20 rows → Update
20 rows → Update

= 5 updates per epoch


## 8. Batch Gradient Descent (BGD)

BGD uses the entire dataset to calculate one overall gradient and then updates w and b once.

10 students:

10 students
↓
Predictions
↓
Individual losses
↓
Overall loss
↓
ONE gradient
↓
ONE update of w and b

If epochs = 3:

Epoch 1 → 10 samples → 1 update
Epoch 2 → 10 samples → 1 update
Epoch 3 → 10 samples → 1 update

Total = 3 updates


## 9. Stochastic Gradient Descent (SGD)

SGD uses one sample at a time to calculate the gradient and immediately update w and b.

Student 1 → Loss → Gradient → Update
Student 2 → Loss → Gradient → Update
Student 3 → Loss → Gradient → Update
...
Student 10 → Loss → Gradient → Update

10 samples + 1 epoch = 10 updates

10 samples + 3 epochs = 30 updates


## 10. Mini-Batch Gradient Descent

Mini-Batch GD uses a small group of samples to calculate one gradient and update w and b.

10 students
Batch size = 2

[1,2] → Gradient → Update
[3,4] → Gradient → Update
[5,6] → Gradient → Update
[7,8] → Gradient → Update
[9,10] → Gradient → Update

1 epoch = 5 updates

If epochs = 3:

5 × 3 = 15 updates


## 11. BGD vs SGD vs Mini-Batch

10 samples:

BGD:
[1 2 3 4 5 6 7 8 9 10] → Update
= 1 update

Mini-Batch:
[1 2] → Update
[3 4] → Update
[5 6] → Update
[7 8] → Update
[9 10] → Update
= 5 updates

SGD:
[1] → Update
[2] → Update
[3] → Update
...
[10] → Update
= 10 updates


## 12. Number of Updates

Updates per epoch:

ceil(Number of samples / Batch size)

Total updates:

Epochs × Updates per epoch

Example:

100 samples
Batch size = 20
Epochs = 3

100 / 20 = 5 updates per epoch

5 × 3 = 15 total updates


## 13. Dataset Does Not Divide Evenly

Example:

100 samples
Batch size = 30

Batch 1 → 30
Batch 2 → 30
Batch 3 → 30
Batch 4 → 10

= 4 updates per epoch


## 14. Loss in Batch Gradient Descent

For 10 students:

Student 1 → Loss₁
Student 2 → Loss₂
Student 3 → Loss₃
...
Student 10 → Loss₁₀

For a mean loss:

Individual losses
↓
Average loss
↓
Overall gradient
↓
Update w and b


## 15. Vectorization

Vectorization = performing mathematical operations on many samples using vectors/matrices instead of Python-level loops.

Without vectorization:

Student 1 → calculation
Student 2 → calculation
Student 3 → calculation
...

With vectorization:

Many samples
↓
Matrix/vector operation
↓
Many results


## 16. Vectorization vs Batch Size

Batch size asks:

"How many samples do I use before one update?"

Vectorization asks:

"How efficiently do I perform calculations on those samples?"

Example:

Mini-Batch:
32 samples → 1 update

Vectorization:
Those 32 samples can be processed using matrix/vector operations.


## 17. Important

Epoch:
Entire dataset processed once.

Batch size:
Number of samples used for one update.

BGD:
Entire dataset → 1 gradient → 1 update.

SGD:
1 sample → 1 gradient → 1 update.

Mini-Batch GD:
Small group → 1 gradient → 1 update.

Vectorization:
Efficient mathematical operations on many samples using vectors/matrices.

Important:
w and b are updated, NOT the individual samples.