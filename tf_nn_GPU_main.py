"""RedLogo PERSONAL tensorflow neural network on GPU project
tf_nn_GPU_main.py -> main file with test data sets"""
import time
import numpy as np
import tensorflow as tf
import tf_nn_GPU_defs as tfnnGPU
from matplotlib import pyplot as plt

__author__ = "RedLogo"
__copyright__ = "copyright 2017, RedLogo's 'PERSONAL tensorflow neural network on GPU' project"
__version__ = "0.1"

# simple X and Y training data initialization
for X_Y_training_data_initialization in range(1):
    # initialize X and Y training data, methodology cited from / refered to cs231n notes from Stanford.
    N = 500  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    hidden_layers = [1000, 400, 200, 100, 50, 20]
    alpha = 0.25
    regularization = 0.0002
    iteration = 5000
    data_type = 64

    X_train_my_format = tfnnGPU.np_zeros(N*K, 2, data_type)
    Y_train_my_format = tfnnGPU.np_zeros(N*K, K, data_type)
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4.5, (j+1)*4.5, N) + np.random.randn(N)*0.4
        X_train_my_format[ix] = np.c_[r*np.cos(1 * t), r*np.sin(1.06 * t)]
        Y_train_my_format[ix, j] = 1
    X_train_my_format = tfnnGPU.np_fp(X_train_my_format, data_type)
    Y_train_my_format = tfnnGPU.np_fp(Y_train_my_format, data_type)

    # plot the training data
    plt.figure(1)
    plt.subplot(121)
    plt.plot(X_train_my_format[Y_train_my_format[:, 0] == 1, 0], X_train_my_format[Y_train_my_format[:, 0] == 1, 1], 'r.')
    plt.plot(X_train_my_format[Y_train_my_format[:, 1] == 1, 0], X_train_my_format[Y_train_my_format[:, 1] == 1, 1], 'b.')
    plt.plot(X_train_my_format[Y_train_my_format[:, 2] == 1, 0], X_train_my_format[Y_train_my_format[:, 2] == 1, 1], 'g.')

    # initialize the prediction X data set
    X_predict = tfnnGPU.np_zeros(200 * 200, 2, data_type)
    for i in range(200):
        for j in range(200):
            X_predict[(i-1)*200 + j, 0] = (i - 100) * 0.01
            X_predict[(i-1)*200 + j, 1] = (j - 100) * 0.01

# initialize all tensors in in GPU, tensorboard is also enabled, some benchmark scalars are saved in tensorboard.
with tf.device('/gpu:0'):
    [X, Y, train_step, accuracy, cross_entropy, final_prediction_thread] = \
        tfnnGPU.tf_relu_and_softmax_layers_init(
            X_train_my_format, Y_train_my_format, hidden_layers, alpha, regularization, data_type
        )
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    # use half of GPU memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("cross_entropy", cross_entropy)
    merged_summary = tf.summary.merge_all()

# call a tf.Session() to execute model training and prediction, with tensorboard enabled, tensorboard graph saved.
with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter('./temp')
    writer.add_graph(sess.graph)
    sess.run(init)
    start_time = time.time()
    for i in range(iteration):
        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict={X: X_train_my_format, Y: Y_train_my_format})
            writer.add_summary(s, i)
        if i % 500 == 0:
            a = sess.run(accuracy, feed_dict={X: X_train_my_format, Y: Y_train_my_format})
            print 'iteration:', i, ', training accuracy is:', a

        sess.run(train_step, feed_dict={X: X_train_my_format, Y: Y_train_my_format})
    run_time = time.time() - start_time
    print 'total run time is:', run_time, 'seconds'
    Y_predict_result = sess.run(final_prediction_thread, feed_dict={X: X_predict})
    Y_predict_result[Y_predict_result > 0.5] = 1
    Y_predict_result[Y_predict_result <= 0.5] = 0

# tensorboard instruction
print 'Further Step 1. in Terminal input: tensorboard --logdir <tensorflow log saving path>'
print 'Further Step 2. then click here: http://127.0.1.1:6006'

# plot the predictions
for plot_prediction_data in range(1):
    plt.subplot(122)
    plt.plot(X_predict[Y_predict_result[:, 0] == 1, 0], X_predict[Y_predict_result[:, 0] == 1, 1], 'rx')
    plt.plot(X_predict[Y_predict_result[:, 1] == 1, 0], X_predict[Y_predict_result[:, 1] == 1, 1], 'bx')
    plt.plot(X_predict[Y_predict_result[:, 2] == 1, 0], X_predict[Y_predict_result[:, 2] == 1, 1], 'gx')
    plt.show()

# run on GTX 1080 Ti (11GB memory, 11TOPS/s capability), tensorflow 1.0.1
# FP16 total run time is: 34.2296459675 seconds
# FP32 total run time is: 25.6236679554 seconds
# FP64 total run time is: 194.550112009 seconds

# run on GTX 1080 Ti (11GB memory, 11TOPS/s capability), tensorflow 1.1.0
# FP16 total run time is: 19.9584898949 seconds
# FP32 total run time is: 14.3683199883 seconds
# FP64 total run time is: 102.75393796 seconds
