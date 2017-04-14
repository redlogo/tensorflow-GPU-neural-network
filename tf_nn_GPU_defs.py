"""RedLogo PERSONAL tensorflow neural network on GPU project
tf_nn_GPU_defs.py -> function definition file"""
import numpy as np
import tensorflow as tf

__author__ = "RedLogo"
__copyright__ = "copyright 2017, RedLogo's 'PERSONAL tensorflow neural network on GPU' project"
__version__ = "0.1"


def np_fp(mat, data_type):
    """np_float documentation:
    transform matrix data type to numpy.float16 or numpy.float32 ro numpy.float64
    input:  mat -> just a matrix
            data_type -> select 16 or 32 or 64
    output: a matrix with same value and with selected data type"""
    if data_type == 16:
        return mat.astype(np.float16, copy=False)
    elif data_type == 32:
        return mat.astype(np.float32, copy=False)
    elif data_type == 64:
        return mat.astype(np.float64, copy=False)
    else:
        print 'Error0: wrong ''data_type'' input to the np_fp function!'
        print 'Error0: select 16 or 32 or 64 only.'
        print 'Error0: however, using numpy.float64 as default for this run.'
        return mat.astype(np.float64, copy=False)


def np_rnd(m, n, data_type):
    """np_rnd documentation:
    give a random matrix with size of (m, n), in data format of numpy.float16 or numpy.float32 or numpy.float64
    input:  m -> row number
            n -> column number
            data_type -> select 16 or 32 or 64
    output: a random matrix with selected data type"""
    return np_fp(np.random.random((m, n)), data_type)


def np_zeros(m, n, data_type):
    """np_zeros documentation:
    give a zeros matrix with size of (m, n), in data format of numpy.float16 or numpy.float32 or numpy.float64
    input:  m -> row number
            n -> column number
            data_type -> select 16 or 32 or 64
    output: a zeros matrix with selected data type"""
    return np_fp(np.zeros((m, n)), data_type)


def np_ones(m, n, data_type):
    """np_ones documentation:
    give a ones matrix with size of (m, n), in data format of numpy.float16 or numpy.float32 or numpy.float64
    input:  m -> row number
            n -> column number
            data_type -> select 16 or 32 or 64
    output: a ones matrix with selected data type"""
    return np_fp(np.ones((m, n)), data_type)


def tf_weight_variable(size, tf_name_scope, data_type):
    """tf_weight_variable documentation:
    define weights in tensorflow format, data format can be tf.float16 or tf.float32 or tf.float64.
        (weights are initialized randomly (stddev=0.15) with tensorflow tf.truncated_normal function.)
    input:  size -> size of matrix, row and col numbers
            tf_name_scope -> tensorflow name scope control
            data_type -> select 16 or 32 or 64
    output: a tensorflow variable with certain size and with selected data type"""
    if data_type == 16:
        return tf.Variable(
            tf.truncated_normal(size, stddev=0.15, dtype=tf.float16), dtype=tf.float16, name=tf_name_scope
        )
    elif data_type == 32:
        return tf.Variable(
            tf.truncated_normal(size, stddev=0.15, dtype=tf.float32), dtype=tf.float32, name=tf_name_scope
        )
    elif data_type == 64:
        return tf.Variable(
            tf.truncated_normal(size, stddev=0.15, dtype=tf.float64), dtype=tf.float64, name=tf_name_scope
        )
    else:
        print 'Error1: wrong ''data_type'' input to the tf_weight_variable function!'
        print 'Error1: select 16 or 32 or 64 only.'
        print 'Error1: however, using tf.float64 as default for this run.'
        return tf.Variable(
            tf.truncated_normal(size, stddev=0.15, dtype=tf.float64), dtype=tf.float64, name=tf_name_scope
        )


def tf_bias_variable(size, tf_name_scope, data_type):
    """tf_bias_variable documentation:
    define bias in tensorflow format, data format can be tf.float16 or tf.float32 or tf.float64.
        (bias are initialized as constant of 0.05 with tensorflow tf.constant function.)
    input:  size -> size of matrix, row and col numbers
            tf_name_scope -> tensorflow name scope control
            data_type -> select 16 or 32 or 64
    output: a tensorflow variable with certain size and with selected data type"""
    if data_type == 16:
        return tf.Variable(
            tf.constant(0.05, shape=size, dtype=tf.float16), dtype=tf.float16, name=tf_name_scope
        )
    elif data_type == 32:
        return tf.Variable(
            tf.constant(0.05, shape=size, dtype=tf.float32), dtype=tf.float32, name=tf_name_scope
        )
    elif data_type == 64:
        return tf.Variable(
            tf.constant(0.05, shape=size, dtype=tf.float64), dtype=tf.float64, name=tf_name_scope
        )
    else:
        print 'Error2: wrong ''data_type'' input to the tf_bias_variable function!'
        print 'Error2: select 16 or 32 or 64 only.'
        print 'Error2: however, using tf.float64 as default for this run.'
        return tf.Variable(
            tf.constant(0.05, shape=size, dtype=tf.float64), dtype=tf.float64, name=tf_name_scope
        )


def tf_relu_layer(input_data, channel_in, channel_out, data_type):
    """tf_relu_layer documentation:
    define ReLU layer in tensorflow format, data format can be tf.float16 or tf.float32 or tf.float64.
    input:  input_data -> previous layer
            channel_in -> how many columns/features/neurons are there in the previous layer
            channel_out -> how many columns/features/neurons are there in this layer
            data_type -> select 16 or 32 or 64
    output: this layer
            L2 regularization thread (for weight) for this layer"""
    with tf.name_scope('RedLogo_ReLU_layer'):
        w = tf_weight_variable([channel_in, channel_out], tf_name_scope='RedLogo_weight', data_type=data_type)
        b = tf_bias_variable([channel_out], tf_name_scope='RedLogo_bias', data_type=data_type)
        w_l2 = tf.nn.l2_loss(w, name='RedLogo_L2_regularization_on_weight')
        layer = tf.nn.relu(tf.matmul(input_data, w) + b, name='RedLogo_relu_operation')
        # tensorboard histogram:
        # tf.summary.histogram('w', w)
        return [layer, w_l2]


def tf_softmax_layer(input_data, channel_in, channel_out, data_type):
    """tf_softmax_layer documentation:
    define Softmax layer in tensorflow format, data format can be tf.float16 or tf.float32 or tf.float64.
    input:  input_data -> previous layer
            channel_in -> how many columns/features/neurons are there in the previous layer
            channel_out -> how many columns/features/neurons/classes are there in this layer
            data_type -> select 16 or 32 or 64
    output: this layer
            L2 regularization thread (for weight) for this layer
            a thread for final prediction using trained neural network structure"""
    with tf.name_scope('RedLogo_softmax_layer'):
        w = tf_weight_variable([channel_in, channel_out], tf_name_scope='RedLogo_weight', data_type=data_type)
        b = tf_bias_variable([channel_out], tf_name_scope='RedLogo_bias', data_type=data_type)
        w_l2 = tf.nn.l2_loss(w, name='RedLogo_L2_regularization_on_weight')
        with tf.name_scope('RedLogo_linear_operation'):
            layer = tf.matmul(input_data, w) + b
        final_prediction_thread = tf.nn.softmax(layer, name='RedLogo_softmax_operation')
        # tensorboard histogram:
        # tf.summary.histogram('w', w)
        return [layer, w_l2, final_prediction_thread]


def tf_relu_and_softmax_layers_init(x, y, hidden_layers, alpha, regularization, data_type):
    """tf_relu_and_softmax_layers_init documentation:
    connect full relu + softmax layers, and initialize cross entropy, train and accuracy measurement sections
    input:  x -> training data x
            y -> traning data y
            hidden_layers -> shape of hidden layers
            alpha -> learning rate for gradient descent method
            regularization -> L2 regularization parameter
            data_type -> select 16 or 32 or 64
    output: X -> tensorflow placeholder
            Y -> tensorflow placeholder
            train_step -> tensorflow train section, need to be called by tf.Session()
            accuracy -> accuracy measurement, need to be called by tf.Session()
            cross_entropy -> cross entropy of the system, need to be called by tf.Session()
            final_prediction_thread -> this is for prediction purpose, prediction is based on trained model"""
    # layer [feature/neuron number per layer] list
    layer_input = list()
    layer_input.append(x.shape[1])
    for item in hidden_layers:
        layer_input.append(item)
    layer_input.append(y.shape[1])
    layer_number = len(layer_input)

    # initialize X and Y training data pipeline, with unfixed rows
    with tf.name_scope('RedLogo_input_data'):
        if data_type == 16:
            X = tf.placeholder(tf.float16, [None, x.shape[1]], name='RedLogo_X_data')
            Y = tf.placeholder(tf.float16, [None, y.shape[1]], name='RedLogo_Y_label')
        elif data_type == 32:
            X = tf.placeholder(tf.float32, [None, x.shape[1]], name='RedLogo_X_data')
            Y = tf.placeholder(tf.float32, [None, y.shape[1]], name='RedLogo_Y_label')
        elif data_type == 64:
            X = tf.placeholder(tf.float64, [None, x.shape[1]], name='RedLogo_X_data')
            Y = tf.placeholder(tf.float64, [None, y.shape[1]], name='RedLogo_Y_label')
        else:
            print 'Error3: wrong ''data_type'' input to the tf_relu_and_softmax_layers_init function!'
            print 'Error3: select 16 or 32 or 64 only.'
            print 'Error3: however, using tf.float64 as default for this run.'
            X = tf.placeholder(tf.float64, [None, x.shape[1]], name='RedLogo_X_data')
            Y = tf.placeholder(tf.float64, [None, y.shape[1]], name='RedLogo_Y_label')

    # initialize layers
    layer = list()
    layer.append(X)
    w_l2 = list()
    for i in range(layer_number - 1):
        if i != layer_number - 2:
            [this_layer, this_w_l2] = tf_relu_layer(
                layer[i], layer_input[i], layer_input[i + 1], data_type
            )
            layer.append(this_layer)
            w_l2.append(this_w_l2)
        else:
            [this_layer, this_w_l2, final_prediction_thread] = tf_softmax_layer(
                layer[i], layer_input[i], layer_input[i + 1], data_type
            )
            layer.append(this_layer)
            w_l2.append(this_w_l2)

    # calculate cross entropy
    with tf.name_scope('RedLogo_cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=Y, logits=layer[-1], name='RedLogo_softmax_cross_entropy'
            )
        )
        for i in range(layer_number - 1):
            cross_entropy = tf.reduce_mean(cross_entropy + regularization * w_l2[i])

    # train section
    with tf.name_scope('RedLogo_train'):
        optimizer = tf.train.GradientDescentOptimizer(alpha)
        train_step = optimizer.minimize(cross_entropy)

    # accuracy measurement section
    with tf.name_scope('RedLogo_accuracy_measurement'):
        correct_prediction = tf.equal(tf.argmax(layer[-1], 1), tf.argmax(Y, 1))
        if data_type == 16:
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))
        elif data_type == 32:
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif data_type == 64:
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        else:
            print 'Error4: wrong ''data_type'' input to the tf_relu_and_softmax_layers_init function!'
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    return [X, Y, train_step, accuracy, cross_entropy, final_prediction_thread]
