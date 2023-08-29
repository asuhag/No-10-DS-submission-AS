import argparse
import os
import tensorflow as tf
import tensorflow.keras as K

# Set up argparse to accept command-line arguments
parser = argparse.ArgumentParser(description='Train a ResNet50 model.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Decay rate.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training.')

# Parse the arguments
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

path_train = 'data/train/'
path_test = 'data/val/'

# Use argparse values
batch_size = args.batch_size
img_height = 640
img_width = 640

train_ds = tf.keras.utils.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    path_train,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    path_test,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred'
)

train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

preprocessing_layer = K.Sequential([
    K.layers.Rescaling(scale=1./127.5, offset=-1, name='Rescaling'),
    K.layers.CenterCrop(img_height, img_width, name='CenterCrop'),
])

data_augmentation = K.Sequential([
    K.layers.RandomFlip("horizontal"),
    K.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    K.layers.RandomRotation(0.2),
    K.layers.RandomContrast(0.2)
])

resnet50 = K.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(640, 640, 3),
    classes=74
)

model = K.models.Sequential([
    K.Input(shape=(640, 640, 3)),
    data_augmentation,
    preprocessing_layer,
    resnet50,
    K.layers.GlobalAveragePooling2D(),
    K.layers.Dense(256, activation='relu'),
    K.layers.Dropout(0.2),
    K.layers.Dense(74, activation='softmax')
])

# Modify the learning schedule with argparse values
lr_schedule = K.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=10000,
    decay_rate=args.decay,
    staircase=True
)

model.compile(
    loss=K.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=K.optimizers.SGD(learning_rate=lr_schedule, momentum=0),
    metrics='accuracy'
)

history = model.fit(
    train_ds,
    epochs=args.epochs,
    validation_data=val_ds,
    batch_size=batch_size
)
