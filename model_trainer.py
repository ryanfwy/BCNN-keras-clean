'''Train model.'''

import os

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.backend import clear_session

from model_builder import buil_bcnn
from data_loader import build_generator

clear_session()

def train_model(
        train_dir,
        valid_dir,
        name_optimizer='sgd',
        learning_rate=1.0,
        decay_learning_rate=1e-8,
        all_trainable=False,
        model_weights_path=None,
        no_class=200,
        batch_size=128,
        epoch=20,

        tensorboard_dir=None,
        checkpoint_dir=None
    ):
    '''Train or retrain model.

    Args:
        train_dir: train dataset directory.
        valid_dir: validation dataset directory.
        name_optimizer: optimizer method.
        learning_rate: learning rate.
        decay_learning_rate: learning rate decay.
        model_weights_path: path of keras model weights.
        no_class: number of prediction classes.
        batch_size: batch size.
        epoch: training epoch.

        tensorboard_dir: tensorboard logs directory.
            If None, dismiss it.
        checkpoint_dir: checkpoints directory.
            If None, dismiss it.

    Returns:
        Training history.
    '''

    model = buil_bcnn(
        all_trainable=all_trainable,
        no_class=no_class,
        name_optimizer=name_optimizer,
        learning_rate=learning_rate,
        decay_learning_rate=decay_learning_rate)

    if model_weights_path:
        model.load_weights(model_weights_path)

    # Load data
    train_generator, valid_generator = build_generator(
        train_dir=train_dir,
        valid_dir=valid_dir,
        batch_size=batch_size)

    # Callbacks
    callbacks = []
    if tensorboard_dir:
        cb_tersoboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=False)
        callbacks.append(cb_tersoboard)

    if checkpoint_dir:
        cb_checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_{epoch:02d}-{val_acc:.3f}.h5'),
            save_weights_only=True,
            monitor='val_loss',
            verbose=True)
        callbacks.append(cb_checkpoint)

    cb_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        min_delta=1e-3)
    cb_stopper = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=10,
        verbose=0,
        mode='auto')
    callbacks += [cb_reducer, cb_stopper]

    # Train
    history = model.fit_generator(
        train_generator,
        epochs=epoch,
        validation_data=valid_generator,
        callbacks=callbacks)

    model.save_weights('./new_model_weights.h5')

    return history


if __name__ == "__main__":
    pass
