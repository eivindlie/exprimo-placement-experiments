import time

import tensorflow as tf
import tensorflow_datasets as tfds

from resnet50 import ResNet50

tfds.disable_progress_bar()


datasets, metadata = tfds.load(
    'imagenette',
    with_info=True,
    as_supervised=True
)

raw_train, raw_validation = datasets['train'], datasets['validation']

get_label_name = metadata.features['label'].int2str

# for image, label in raw_train.take(5):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))
#     plt.show()

IMG_SIZE = 224


def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, include_top=True, weights=None, classes=1000)

model = base_model

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(),
)

batch_times = []

NUM_BATCHES = 50 + 1

print('Starting training')
i = 0
for image_batch, label_batch in train_batches:
    print(f'Batch {i + 1}/{NUM_BATCHES}...', end='')
    start_time = time.clock()
    model.train_on_batch(image_batch, label_batch)
    end_time = time.clock()

    elapsed_time = (end_time - start_time) * 1000
    print(f'\t{elapsed_time}ms')
    batch_times.append(elapsed_time)

    i += 1
    if i >= NUM_BATCHES:
        break

print(f'Batch times: {batch_times}')

#history = model.fit(train_batches, epochs=1, validation_data=validation_batches)
