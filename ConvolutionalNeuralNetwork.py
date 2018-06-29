import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


def one_hot(labels, num_classes):
    results = np.zeros(shape=(len(labels), num_classes), dtype=np.float32)
    for i, values in enumerate(labels):
        results[i,values] = 1.
    return results


input_l = tf.keras.Input((28,28, 1))
conv_1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input_l)
maxpool_1 = tf.keras.layers.MaxPooling2D((2,2))(conv_1)
conv_2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(maxpool_1)
maxpool_2 = tf.keras.layers.MaxPooling2D((2,2))(conv_2)
conv_3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(maxpool_2)
flatten = tf.keras.layers.Flatten()(conv_3)
dense_1 = tf.keras.layers.Dense(64, activation='relu')(flatten)
dense_2 = tf.keras.layers.Dense(10, activation='softmax')(dense_1)

model = tf.keras.Model(inputs=[input_l], outputs=[dense_2])
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
print(model.summary())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print('shape training images ', train_images.shape)
print('shape test images ', test_images.shape)
train_images_reshaped = train_images.reshape((-1, 28, 28, 1))
test_images_reshaped = test_images.reshape((-1, 28, 28, 1))

train_labels_one_hot = one_hot(train_labels, 10)
test_labels_one_hot = one_hot(test_labels, 10)

history = model.fit(train_images_reshaped, train_labels_one_hot, validation_split=0.1, epochs=5, batch_size=64);

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate(test_images_reshaped, test_labels_one_hot)
print('test loss={}, accuracy={}'.format(test_loss, test_acc))