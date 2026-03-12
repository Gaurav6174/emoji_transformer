import os
import tensorflow as tf
from config import MODEL_DIR

def get_callbacks() -> list:
    
    """
    callbacks: 
        1.modelcheckpoint
        2. early stopping
        3. reduce learning plateau
        4. CSVLogger
        5. TensorBoard
    """
    
    os.makedirs(MODEL_DIR, exist_ok = True)
    os.makedirs(os.path.join(MODEL_DIR, "logs"), exist_ok = True)
    
    callbacks = [
        _model_checkpoint(),
        _early_stopping(),
        #_reduce_lr(),
        _csv_logger(),
        _tensorboard(),
    ]

    return callbacks

def _model_checkpoint() -> tf.keras.callbacks.ModelCheckpoint:
    path = os.path.join(MODEL_DIR, "best_weights.weights.h5")

    return tf.keras.callbacks.ModelCheckpoint(
        filepath          = path,
        monitor           = "val_accuracy",
        save_best_only    = True,
        save_weights_only = True,
        mode              = "max",
        verbose           = 1
    )
    
def _early_stopping() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor              = "val_loss",
        patience             = 10,
        restore_best_weights = True,
        min_delta            = 1e-4,
        verbose              = 1
    )
    
def _reduce_lr() -> tf.keras.callbacks.ReduceLROnPlateau:
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-6,
        cooldown = 1,
        verbose  = 1
    )
    
def _csv_logger() -> tf.keras.callbacks.CSVLogger:
    path = os.path.join(MODEL_DIR, "training_log.csv")

    return tf.keras.callbacks.CSVLogger(
        filename = path,
        append   = False    # overwrite on each new training run
    )
    
def _tensorboard() -> tf.keras.callbacks.TensorBoard:
    
    #After training starts, open a new terminal and run:
    #    tensorboard --logdir saved_model/logs

    log_dir = os.path.join(MODEL_DIR, "logs")

    return tf.keras.callbacks.TensorBoard(
        log_dir       = log_dir,
        histogram_freq  = 1,
        update_freq     = "epoch"
    )

    

    