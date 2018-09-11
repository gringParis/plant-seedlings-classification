
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from datetime import datetime
import argparse
# local import
import data_preparation
import models

parser = argparse.ArgumentParser(
    description='Train your model for plant identification.')
parser.add_argument('-p',  type=str, default='/Volumes/my-hard-drive/ml-data/plant-seedlings-classification/',
                    help='path to the data ')
parser.add_argument('-m',  type=str, default='v1',
                    help=' version of the model to call ')
parser.add_argument('-s',  type=str, default='128',
                    help=' size of the image to work with ')


def train(path, model_version, image_size):
    # main hyperparameters
    lr = 0.001
    epochs = 100
    mini_batch_size = 128
    validation_minibatch_size = 64
    if model_version == 'v1':
        model_call = models.v1
    elif model_version == 'v2':
        model_call = models.v2
    elif model_version == 'v3':
        model_call = models.v3
    model_name = '4conv_{}_lr_{}_fcunits_4096_epochs_{}'.format(model_version,
                                                                lr, epochs)

    # get prepared data generators
    datagen, valgen, x_train, y_train, x_val, y_val = data_preparation.get_training_data(
        path, preprocess_path='preprocessed-{}'.format(image_size))
    nb_minibatches = int(x_train.shape[0] / mini_batch_size) + 1
    val_nb_minibatches = int(x_val.shape[0] / validation_minibatch_size) + 1
    # X, Y, is_training, cost, acc, opt = models.get_4conv_3d(
    #     x_train, y_train, lr)
    X, Y, Class_weights, is_training, cost, acc, opt = model_call(
        x_train, y_train, lr)
    full_y = np.argmax(np.vstack((y_train, y_val)), axis=1)
    classes = np.unique(full_y)
    print(full_y.shape, classes)
    class_weights = class_weight.compute_class_weight(
        'balanced', classes, full_y).reshape((y_train.shape[1], 1))

    # .reshape((y_train.shape[1], 1))
    with tf.name_scope("visualisation"):
        # reset the date for tensorboard
        now = datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S")
        root_logdir = "logs"
        logdir = "{}/run-{}/".format(root_logdir, now)
        saver = tf.train.Saver()
        cost_summary = tf.summary.scalar('cost', cost)
        val_cost_summary = tf.summary.scalar('val_cost', cost)
        accuracy_summary = tf.summary.scalar('acc', acc)
        validation_accuracy_summary = tf.summary.scalar('val_acc', acc)
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        # init variables
        sess.run(tf.global_variables_initializer())
        cp = 0
        val_cp = 0
        for epoch in range(epochs):
            print("starting epoch ", epoch, "minibatches: ", nb_minibatches)
            epoch_cost = 0
            mini_i = 0
            timeit = datetime.now()

            for minibatch_x, minibatch_y in datagen.flow(x_train, y_train, batch_size=mini_batch_size):
                # training
                _, batch_cost = sess.run([opt, cost], feed_dict={is_training: True,
                                                                 X: minibatch_x,
                                                                 Y: minibatch_y,
                                                                 Class_weights: class_weights})
                epoch_cost += batch_cost
                cp += 1
                mini_i += 1
                # display and check metrics
                if mini_i % 1 == 0 and not(mini_i == 0):
                    # calculate remaining time in epoch
                    delta = datetime.now() - timeit
                    remaining = (delta * nb_minibatches / mini_i) - delta
                    show_remaining = remaining.total_seconds()
                    epoch_cp = (cp * 100) / nb_minibatches
                    # add metrics to tensorboard
                    summary_str1 = accuracy_summary.eval(
                        feed_dict={is_training: True, X: minibatch_x, Y: minibatch_y, Class_weights: class_weights})
                    file_writer.add_summary(summary_str1, epoch_cp)
                    summary_str2 = cost_summary.eval(
                        feed_dict={is_training: True, X: minibatch_x, Y: minibatch_y, Class_weights: class_weights})
                    file_writer.add_summary(summary_str2, epoch_cp)

                    print("after ", epoch_cp * 1. / 100,  "epochs, cost ", str(
                        epoch_cost/(mini_i + 1)))
                    print("remaining in epoch : ", show_remaining, " s")
                if mini_i >= nb_minibatches:
                    break
            mini_i = 0
            vc = 0
            for val_x, val_y in valgen.flow(x_val, y_val, batch_size=validation_minibatch_size):
                if mini_i >= val_nb_minibatches:
                    break
                val_epoch_cp = (val_cp * 100) / val_nb_minibatches
                # add validation metrics
                sum2 = validation_accuracy_summary.eval(
                    feed_dict={is_training: False, X: val_x, Y: val_y, Class_weights: class_weights})
                sum1 = val_cost_summary.eval(
                    feed_dict={is_training: False, X: val_x, Y: val_y, Class_weights: class_weights})
                file_writer.add_summary(sum2, val_epoch_cp)
                file_writer.add_summary(sum1, val_epoch_cp)
                mini_batch_val_cost = cost.eval(
                    feed_dict={is_training: False, X: val_x, Y: val_y, Class_weights: class_weights})
                vc += mini_batch_val_cost
                mini_i += 1
                val_cp += 1
                print("minibatch validation cost : ", mini_batch_val_cost)
            vc = vc / val_nb_minibatches
            print("epoch_cost : ", str(epoch_cost /
                                       nb_minibatches),  " epoch_val_cost :", str(vc))
            _ = saver.save(sess, os.getcwd() +
                           "/tmp/" + model_name + ".ckpt")


if __name__ == "__main__":
    print('start main')
    args = parser.parse_args()
    train(args.p, args.m, args.s)
    # Training logic

    # Test logic
    # test()
