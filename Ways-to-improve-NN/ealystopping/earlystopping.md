#              EARLY STOPPING


Purpose:
    Prevent overfitting by stopping training when
    validation performance stops improving.

--------------------------------------------------
PARAMETERS
--------------------------------------------------

1. monitor
   What metric should be monitored?

   "val_loss"       -> monitor validation loss
   "val_accuracy"   -> monitor validation accuracy

--------------------------------------------------

2. min_delta
   Minimum improvement required to count as
   an actual improvement.

   min_delta = 0
   -> Any improvement counts

   min_delta = 0.001
   -> Improvement must be at least 0.001

--------------------------------------------------

3. patience
   Number of epochs to wait without improvement
   before stopping.

   patience = 10
   -> Wait 10 epochs without improvement

--------------------------------------------------

4. verbose
   Controls the stopping message.

   verbose = 0
   -> No message

   verbose = 1
   -> Show "early stopping" message

--------------------------------------------------

5. mode
   Determines whether higher or lower is better.

   "min"  -> Lower is better
             Example: val_loss

   "max"  -> Higher is better
             Example: val_accuracy

   "auto"  -> Keras determines automatically

--------------------------------------------------

6. baseline
   Minimum/expected performance level.

   None
   -> No baseline specified

--------------------------------------------------

7. restore_best_weights
   What weights should be kept after stopping?

   False -> Keep final weights

   True  -> Restore weights from the epoch
            with the best monitored value

--------------------------------------------------

8. start_from_epoch
   When should EarlyStopping start monitoring?

   0  -> Start from the beginning

   10 -> Start monitoring after epoch 10


# BEST SIMPLE SETUP

EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    mode="min",
    restore_best_weights=True
)
