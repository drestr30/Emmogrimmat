# Keras center and random crop support for ImageDataGenerator

`preprocess_crop.py` script below adds center and random crop to Keras's `flow_from_directory` data generator.

It first resizes image preserving aspect ratio and then performs crop. Resized image size is based on `crop_fraction` which is hardcoded but can be changed. See `crop_fraction = 0.875` line where 0.875 appears to be the most common, e.g. 224px crop from 256px image.

Note that the implementation has been done by monkey patching `keras_preprocessing.image.utils.loag_img` function as I couldn't find any other way to perform crop before resizing without rewriting many other classes above.

Due to these limitations, the cropping method is enumerated into the `interpolation` field. Methods are delimited by `:` where the first part is interpolation and second is crop e.g. `lanczos:random`. Supported crop methods are `none`, `center`, `random`. When no crop method is specified, `none` is assumed.


## How to use it

Just drop the `preprocess_crop.py` into your project to enable cropping. The example below shows how you can use random cropping for the training and center cropping for validation:

```python
import preprocess_crop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

#...

# Training with random crop

train_datagen = ImageDataGenerator(
    rotation_range=20,
    channel_shift_range=20,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

train_img_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    interpolation = 'lanczos:random', # <--------- random crop
    shuffle = True
)

# Validation with center crop

validate_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

validate_img_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    interpolation = 'lanczos:center', # <--------- center crop
    shuffle = False
)
```