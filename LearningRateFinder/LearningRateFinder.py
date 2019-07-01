# coding:utf-8

import numpy as np
import keras.backend as K
from tensorflow import set_random_seed
from keras.callbacks import LambdaCallback
np.random.seed(7)
set_random_seed(7)


class LearningRateFinder(object):

    def __init__(self, net, lr_start=1e-6, lr_multiplier=1.1, smoothing=0.3):
        # init parameter
        self.__net = net
        self.__weights = self.__net.get_weights()
        self.__lr_start = lr_start
        self.__lr_multiplier = lr_multiplier

        # stop criteria
        self.__counter = 0
        self.__smoothing = smoothing
        self.__first_loss = None
        self.__exponential_moving_average = None

        # log
        self.__lrs = []
        self.__losses = []

    def find_lr(self, trn_feature, trn_label):
        def on_batch_end(logs):
            self.__counter += 1

            lr = K.get_value(self.__net.optimizer.lr)
            self.__lrs.append(lr)

            loss = logs["loss"]
            self.__losses.append(loss)

            if self.__first_loss is None:
                self.__first_loss = loss

            if self.__exponential_moving_average is None:
                self.__exponential_moving_average = loss
            else:
                self.__exponential_moving_average = \
                    ((1 - self.__smoothing) * loss) + (self.__smoothing * self.__exponential_moving_average)

            if self.__exponential_moving_average > self.__first_loss * 2 and self.__counter >= 20:
                self.__net.stop_training = True
                return

            lr *= self.__lr_multiplier
            K.set_value(self.__net.optimizer.lr, lr)

        K.set_value(self.__net.optimizer.lr, self.__lr_start)
        self.__net.fit(
            trn_feature,
            trn_label,
            batch_size=2046,
            verbose=2,
            epochs=999999,
            callbacks=[LambdaCallback(on_batch_end=lambda _, logs: on_batch_end(logs))]
        )

    def get_net(self):
        self.__net.set_weights(self.__weights)
        K.set_value(self.__net.optimizer.lr, min(self.__lrs[self.__losses.index(min(self.__losses))], 0.001))

        print("\n")
        print("*" * 36)
        print(K.get_value(self.__net.optimizer.lr))
        print("*" * 36)

        return self.__net
