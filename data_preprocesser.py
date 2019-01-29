'''Preprocessing data.'''

import os
import numpy as np
import cv2

from tensorflow.python.keras.preprocessing.image import DirectoryIterator as Keras_DirectoryIterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator as Keras_ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.python.keras.backend import floatx

np.random.seed(3)

def resize_image(
        x,
        size_target=(448, 448),
        rate_scale=1.0,
        flg_keep_aspect=False,
        flg_random_scale=False):
    '''Resizing image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.
        rate_scale: scale rate.
        flg_keep_aspect: a bool of keeping image aspect or not.
        flg_random_scale: a bool of scaling image randomly.

    Returns:
        Resized image.
    '''

    # Convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Calculate resize coefficients
    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c , = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c , = img.shape

    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]
    if size_target is None:
        size_heigth_target = size_height_img * rate_scale
        size_width_target = size_width_img * rate_scale

    coef_height, coef_width = 1, 1
    if size_height_img < size_heigth_target:
        coef_height = size_heigth_target / size_height_img
    if size_width_img < size_width_target:
        coef_width = size_width_target / size_width_img

    # Calculate coeffieient to match small size to target size
    ## scale coefficient if specified
    low_scale = rate_scale
    if flg_random_scale:
        low_scale = 1.0
    coef_max = max(coef_height, coef_width) * np.random.uniform(low=low_scale, high=rate_scale)

    # Resize image
    size_height_resize = np.ceil(size_height_img*coef_max)
    size_width_resize = np.ceil(size_width_img*coef_max)

    method_interpolation = cv2.INTER_CUBIC

    if flg_keep_aspect:
        img_resized = cv2.resize(
            img,
            dsize=(int(size_width_resize), int(size_height_resize)),
            interpolation=method_interpolation)
    else:
        img_resized = cv2.resize(
            img,
            dsize=(
                int(size_width_target*np.random.uniform(low=low_scale, high=rate_scale)),
                int(size_heigth_target*np.random.uniform(low=low_scale, high=rate_scale))),
            interpolation=method_interpolation)

    return img_resized

def center_crop_image(x, size_target=(448, 448)):
    '''Crop image from center point.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Center cropped image.
    '''

    # Convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Set size
    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c, = img.shape

    # Crop image
    h_start = int((size_height_img - size_heigth_target) / 2)
    w_start = int((size_width_img - size_width_target) / 2)
    img_cropped = img[h_start:h_start+size_heigth_target, w_start:w_start+size_width_target, :]

    return img_cropped

def random_crop_image(x, size_target=(448, 448)):
    '''Crop image from random point.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Random cropped image.
    '''

    # Convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Set size
    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c , = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c , = img.shape

    # Crop image
    margin_h = (size_height_img - size_heigth_target)
    margin_w = (size_width_img - size_width_target)
    h_start = 0 
    w_start = 0
    if margin_h != 0:
        h_start = np.random.randint(low=0, high=margin_h)
    if margin_w != 0:
        w_start = np.random.randint(low=0, high=margin_w)
    img_cropped = img[h_start:h_start+size_heigth_target, w_start:w_start+size_width_target, :]

    return img_cropped

def horizontal_flip_image(x):
    '''Flip image horizontally.

    Args:
        x: input image.

    Returns:
        Horizontal flipped image.
    '''

    if np.random.random() >= 0.5:
        return x[:, ::-1, :]
    else:
        return x

def normalize_image(x, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)):
    '''Normalization.

    Args:
        x: input image.
        mean: mean value of the input image.
        std: standard deviation value of the input image.

    Returns:
        Normalized image.
    '''

    x = np.asarray(x, dtype=np.float32)
    if len(x.shape) == 4:
        for dim in range(3):
            x[:, :, :, dim] = (x[:, :, :, dim] - mean[dim]) / std[dim]
    if len(x.shape) == 3:
        for dim in range(3):
            x[:, :, dim] = (x[:, :, dim] - mean[dim]) / std[dim]

    return x

def preprocess_input(x):
    '''Preprocesses a tensor or Numpy array encoding a batch of images.'''

    return normalize_image(x, mean=[123.82988033, 127.3509729, 110.25606303])


class DirectoryIterator(Keras_DirectoryIterator):
    '''Inherit from keras' DirectoryIterator'''
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=floatx())
        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(
                os.path.join(self.directory, fname),
                grayscale=grayscale,
                target_size=None,
                interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)

            # Pillow images should be closed after `load_img`, but not PIL images.
            if hasattr(img, 'close'):
                img.close()

            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # Optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # Build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        return batch_x, batch_y


class ImageDataGenerator(Keras_ImageDataGenerator):
    '''Inherit from keras' ImageDataGenerator'''
    def flow_from_directory(
            self, directory,
            target_size=(256, 256), color_mode='rgb',
            classes=None, class_mode='categorical',
            batch_size=16, shuffle=True, seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            follow_links=False,
            subset=None,
            interpolation='nearest'
        ):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


if __name__ == "__main__":
    pass
