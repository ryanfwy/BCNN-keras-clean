'''Load data and build generators.'''

from data_preprocesser import normalize_image, random_crop_image, center_crop_image
from data_preprocesser import resize_image, horizontal_flip_image
from data_preprocesser import ImageDataGenerator

def train_preprocessing(x, size_target=(448, 448)):
    '''Preprocessing for train dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return normalize_image(
        random_crop_image(
            horizontal_flip_image(
                resize_image(
                    x,
                    size_target=size_target,
                    flg_keep_aspect=True
                )
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )

def valid_preprocessing(x, size_target=(448, 448)):
    '''Preprocessing for validation dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return normalize_image(
        center_crop_image(
            resize_image(
                x,
                size_target=size_target,
                flg_keep_aspect=True
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )

def build_generator(
        train_dir=None,
        valid_dir=None,
        batch_size=128
    ):
    '''Build train and validation dataset generators.

    Args:
        train_dir: train dataset directory.
        valid_dir: validation dataset directory.
        batch_size: batch size.

    Returns:
        Train generator and validation generator.
    '''
    results = []
    if train_dir:
        train_datagen = ImageDataGenerator(
            preprocessing_function=train_preprocessing
        )
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(448, 448),
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [train_generator]

    if valid_dir:
        valid_datagen = ImageDataGenerator(
            preprocessing_function=valid_preprocessing
        )
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(448, 448),
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [valid_generator]

    return results


if __name__ == "__main__":
    pass
