import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils


def dense(input_tensor, output_units, scope, activation=None):
    with tf.name_scope(scope):
        #shape of the weights matrix
        shape = (input_tensor.shape.as_list()[1], output_units)
        # weights matrix
        weights = tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32), name='W')
        # bias vector (**brodcast**)
        bias = tf.Variable(tf.zeros(shape=[output_units], dtype=tf.float32), name='b')
        # define the layers as W * x + b
        layer = tf.add(tf.matmul(input_tensor, weights), bias)
        # add the squashing function (non linearity)
        if activation is not None:
            return activation(layer)
        else:
            return layer


tf.logging.set_verbosity(tf.logging.INFO)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print('shape training images ', train_images.shape)
print('shape test images ', test_images.shape)

num_classes = 10

i = np.random.randint(0, len(train_images))
print('displaying image {} with class {}'.format(i, train_labels[i]))
plt.imshow(train_images[i], cmap='gray');
plt.show()

g = tf.Graph()

plt.show()

flatten_shape = np.prod(train_images.shape[1:])

with g.as_default():
    X = tf.placeholder(tf.float32, [None, flatten_shape], name='X')
    y = tf.placeholder(tf.float32, [None, 10], name='y')

utils.show_graph(g)

with g.as_default():
    # define the model
    l1 = dense(X, 32, 'h1', activation=tf.nn.sigmoid)
    l2 = dense(l1, 64, 'h2', activation=tf.nn.relu)
    logits = dense(l1, 10, 'out', activation=None)

    # define the loss function
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=y))

    # define the optimizer
    optmizer = tf.train.RMSPropOptimizer(learning_rate=0.01)

    # train operation
    train_op = optmizer.minimize(loss_op)

    # metrics
    correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

utils.show_graph(g)

epochs = 10
batch_size = 128
val_step = int(len(train_images) * 0.9)

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

#reshape the images
train_images_reshaped = np.reshape(train_images, (len(train_images), -1))
test_images_reshaped = np.reshape(test_images, [len(test_images), -1])

print('shape train images {}'.format(train_images_reshaped.shape))


def one_hot(labels, num_classes):
    results = np.zeros(shape=(len(labels), num_classes), dtype=np.float32)
    for i, values in enumerate(labels):
        results[i,values] = 1.
    return results


train_labels_one_hot = one_hot(train_labels, 10)
test_labels_one_hot = one_hot(test_labels, 10)
print('shape train labels with one hot econding {}'.format(train_labels_one_hot.shape))

train_x, val_x = train_images_reshaped[:val_step], train_images_reshaped[val_step:]
train_y, val_y = train_labels_one_hot[:val_step], train_labels_one_hot[val_step:]

num_batches = len(train_x) // batch_size


def to_batch(x,y, batch_size, shuffle=True):
    idxs = np.arange(0, len(x))
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]
    num_batches = len(x) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        if  end < len(x):
            yield x[start:end], y[start:end]
        else:
            yield x[start:], y[start:]


with tf.Session(graph=g) as sess:
    # initialize variables (i.e. assign to their default value)
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        gen = to_batch(train_x, train_y, batch_size)
        for _ in range(num_batches):
            x_batch, y_batch = next(gen)
            sess.run(train_op, feed_dict={X: x_batch, y: y_batch})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: val_x, y: val_y})
        print("Epochs {}, val_loss={:.3f}, val_acc={:.3f}".format(e, loss, acc))

    print('training finished')

    test_acc = sess.run(accuracy, feed_dict={X: test_images_reshaped, y: test_labels_one_hot})
    print('Testing accuracy {}'.format(test_acc))

    saver = tf.train.Saver()
    save_path = saver.save(sess, '/tmp/model.ckpt')
    print('model saved in path {}'.format(save_path))

with tf.Session(graph=g) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, '/tmp/model.ckpt')

    i = np.random.randint(0, len(test_images))
    res = np.argmax(sess.run(logits, feed_dict={X: [test_images_reshaped[i]]}))
    print('class predicted {}, expected {}'.format(res, test_labels[i]))
    plt.imshow(test_images[i], cmap='gray')