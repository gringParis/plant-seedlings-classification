import tensorflow as tf
from functools import partial


def v1(x, y, lr):
    """
    Simple cnn model 
    """
    tf.reset_default_graph()
    # create placeholders
    X = tf.placeholder(tf.float32, shape=(
        None, x.shape[1], x.shape[2], x.shape[3]))
    Y = tf.placeholder(tf.float32, shape=(None, y.shape[1]))
    # for batchnorm
    is_training = tf.placeholder_with_default(
        False, shape=(), name='is_training')
    # for class weights
    Class_weights = tf.placeholder(tf.float32, shape=(None, 1))

    w_parameter = partial(
        tf.get_variable, initializer=tf.contrib.layers.xavier_initializer(seed=0))

    conv = partial(tf.nn.conv2d, strides=[1, 1, 1, 1], padding='SAME')
    maxpool = partial(tf.nn.max_pool, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='VALID')
    # initialize weights
    W1 = w_parameter('W1', shape=(3, 3, 3, 32))
    W2 = w_parameter('W2', shape=(3, 3, 32, 64))
    W3 = w_parameter('W3', shape=(3, 3, 64, 128))
    W4 = w_parameter('W4', shape=(3, 3, 128, 256))

    with tf.name_scope("model"):
        # define computational graph
        # convolution 1
        Z1 = conv(X, W1)
        A1 = tf.nn.relu(Z1)
        P1 = maxpool(A1)
        # convolution 2
        Z2 = conv(P1, W2)
        A2 = tf.nn.relu(Z2)
        P2 = maxpool(A2)
        # convolution 3
        Z3 = conv(P2, W3)

        A3 = tf.nn.relu(Z3)
        P3 = maxpool(A3)
        # convolution 4
        Z4 = conv(P3, W4)
        A4 = tf.nn.relu(Z4)
        P4 = maxpool(A4)

        # fully connected
        P4 = tf.contrib.layers.flatten(P4)
        A5 = tf.contrib.layers.fully_connected(P4, 4096)
        A5 = tf.layers.dropout(A5)
        A6 = tf.contrib.layers.fully_connected(A5, 4096)
        A6 = tf.layers.dropout(A6)
        # fully connected
        Z6 = tf.contrib.layers.fully_connected(
            A6, y.shape[1], activation_fn=None)

    with tf.name_scope("loss"):
        # cost
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z6))
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Z6, 1)), 'float'), name='acc')

    with tf.name_scope("train"):
        # optimisation
        opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    return X, Y,  Class_weights, is_training, cost, acc, opt


def v2(x, y, lr):
    """
     Cnn with batch normalization
     """
    tf.reset_default_graph()
    # create placeholders
    X = tf.placeholder(tf.float32, shape=(
        None, x.shape[1], x.shape[2], x.shape[3]))
    Y = tf.placeholder(tf.float32, shape=(None, y.shape[1]))
    # for batchnorm
    is_training = tf.placeholder_with_default(
        False, shape=(), name='is_training')
    # for class weights
    Class_weights = tf.placeholder(tf.float32, shape=(None, 1))
    # repetitiv parts of model definition
    w_parameter = partial(
        tf.get_variable, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    dense = partial(
        tf.layers.dense, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    conv = partial(tf.nn.conv2d, strides=[1, 1, 1, 1], padding='SAME')
    maxpool = partial(tf.nn.max_pool, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='VALID')
    batch_norm = partial(tf.layers.batch_normalization,
                         training=is_training, momentum=0.9)

    # initialize weights
    W1 = w_parameter('W1', shape=(3, 3, 3, 32))
    W2 = w_parameter('W2', shape=(3, 3, 32, 64))
    W3 = w_parameter('W3', shape=(3, 3, 64, 128))
    W4 = w_parameter('W4', shape=(3, 3, 128, 256))

    with tf.name_scope("model"):
        # define computational graph
        # convolution 1
        Z1 = conv(X, W1)
        Z1 = batch_norm(Z1)
        A1 = tf.nn.relu(Z1)
        P1 = maxpool(A1)
        # convolution 2
        Z2 = conv(P1, W2)
        Z2 = batch_norm(Z2)
        A2 = tf.nn.relu(Z2)
        P2 = maxpool(A2)
        # convolution 3
        Z3 = conv(P2, W3)
        Z3 = batch_norm(Z3)
        A3 = tf.nn.relu(Z3)
        P3 = maxpool(A3)

        # convolution 4
        Z4 = conv(P3, W4)
        Z4 = batch_norm(Z4)
        A4 = tf.nn.relu(Z4)
        P4 = maxpool(A4)

        # fully connected
        P4 = tf.contrib.layers.flatten(P4)
        Z5 = dense(P4, 4096)
        A5 = tf.nn.relu(Z5)
        A5 = tf.layers.dropout(A5)
        Z6 = dense(A5, 4096)
        A6 = tf.nn.relu(Z6)
        A6 = tf.layers.dropout(A6)
        # fully connected
        Z7 = tf.contrib.layers.fully_connected(
            A6, y.shape[1], activation_fn=None)

    with tf.name_scope("loss"):
        # cost
        cost = tf.losses.softmax_cross_entropy(Y, Z7)

        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Z7, 1)), 'float'), name='acc')

    with tf.name_scope("train"):
        # optimisation
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # part relative to batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            training_op = opt.minimize(cost)
    return X, Y,  Class_weights, is_training, cost, acc, training_op


def v3(x, y, lr):
    """
     Cnn with batch normalization and class weighted loss
     """
    tf.reset_default_graph()
    # create placeholders
    X = tf.placeholder(tf.float32, shape=(
        None, x.shape[1], x.shape[2], x.shape[3]))
    Y = tf.placeholder(tf.float32, shape=(None, y.shape[1]))
    Class_weights = tf.placeholder(tf.float32, shape=(None, 1))
    # for batchnorm
    is_training = tf.placeholder_with_default(
        False, shape=(), name='is_training')

    # repetitiv parts of model definition
    w_parameter = partial(
        tf.get_variable, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    dense = partial(
        tf.layers.dense, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
    conv = partial(tf.nn.conv2d, strides=[1, 1, 1, 1], padding='SAME')
    maxpool = partial(tf.nn.max_pool, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='VALID')
    batch_norm = partial(tf.layers.batch_normalization,
                         training=is_training, momentum=0.9)

    # initialize weights
    W1 = w_parameter('W1', shape=(3, 3, 3, 32))
    W2 = w_parameter('W2', shape=(3, 3, 32, 64))
    W3 = w_parameter('W3', shape=(3, 3, 64, 128))
    W4 = w_parameter('W4', shape=(3, 3, 128, 256))
    W5 = w_parameter('W5', shape=(3, 3, 256, 512))

    with tf.name_scope("model"):
        # define computational graph
        # convolution 1
        Z1 = conv(X, W1)
        Z1 = batch_norm(Z1)
        A1 = tf.nn.relu(Z1)
        P1 = maxpool(A1)
        # convolution 2
        Z2 = conv(P1, W2)
        Z2 = batch_norm(Z2)
        A2 = tf.nn.relu(Z2)
        P2 = maxpool(A2)
        # convolution 3
        Z3 = conv(P2, W3)
        Z3 = batch_norm(Z3)
        A3 = tf.nn.relu(Z3)
        P3 = maxpool(A3)

        # convolution 4
        Z4 = conv(P3, W4)
        Z4 = batch_norm(Z4)
        A4 = tf.nn.relu(Z4)
        P4 = maxpool(A4)

        # convolution 5
        Z4b = conv(P4, W5)
        Z4b = batch_norm(Z4b)
        A4b = tf.nn.relu(Z4b)
        P4b = maxpool(A4b)

        # fully connected
        P4 = tf.contrib.layers.flatten(P4b)
        Z5 = dense(P4, 2048)
        Z5 = batch_norm(Z5)
        A5 = tf.nn.relu(Z5)
        A5 = tf.layers.dropout(A5)
        Z6 = dense(A5, 2048)
        Z6 = batch_norm(Z6)
        A6 = tf.nn.relu(Z6)
        A6 = tf.layers.dropout(A6)
        # fully connected
        Z7 = tf.contrib.layers.fully_connected(
            A6, y.shape[1], activation_fn=None)

    with tf.name_scope("loss"):
        # calculate weight of each example based on it class
        weights = tf.matmul(Y, Class_weights)
        weights = tf.reshape(weights, [-1])
        # cost
        cost = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(Y, Z7, weights=weights, reduction=tf.losses.Reduction.NONE))

        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Z7, 1)), 'float'), name='acc')

    with tf.name_scope("train"):
        # optimisation
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # part relative to batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            training_op = opt.minimize(cost)
    return X, Y, Class_weights, is_training, cost, acc, training_op
