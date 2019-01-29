'''Main script.'''

import os
from model_trainer import train_model

def transfer_learning(
        train_dir,
        valid_dir,
        no_class=200,
        epoch=10,
        batch_size=128,
        tensorboard_dir=None,
        checkpoint_dir=None
    ):
    '''For transfer learning.

    Build the model and fix all VGG16 layers to untrainable.
    Transfer the FC layer to the target classes.
    '''

    return train_model(
        name_optimizer='adam',
        learning_rate=0.001,
        decay_learning_rate=1e-9,
        train_dir=train_dir,
        valid_dir=valid_dir,
        no_class=no_class,
        epoch=epoch,
        batch_size=batch_size,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir
    )

def fine_tuning(
        train_dir,
        valid_dir,
        model_weights_path,
        all_trainable=True,
        no_class=200,
        epoch=20,
        batch_size=128,
        tensorboard_dir=None,
        checkpoint_dir=None
    ):
    '''For fine tuning.

    Load a model and make all layers trainbale.
    Fine tune the whole model.
    '''

    return train_model(
        learning_rate=0.0001,
        decay_learning_rate=1e-8,
        train_dir=train_dir,
        valid_dir=valid_dir,
        all_trainable=all_trainable,
        model_weights_path=model_weights_path,
        no_class=no_class,
        epoch=epoch,
        batch_size=batch_size,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir
    )


if __name__ == "__main__":
    pass
