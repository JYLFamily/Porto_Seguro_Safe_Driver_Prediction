# coding:utf-8

import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Dropout, AlphaDropout, Concatenate, Dense
from keras.initializers import random_uniform, lecun_normal, constant
from scipy.special import logit
from tensorflow import set_random_seed
np.random.seed(7)
set_random_seed(7)


def gini(y_true, y_pred):
    num = len(y_true)
    a_c = y_true[np.argsort(y_pred)].cumsum()

    return (a_c.sum() / a_c[-1] - (num + 1) / 2.0) / num


def gini_normalized(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)


def network(categorical_columns_item, num_deep_numeric_feature, num_wide_numeric_feature, bias):
    input_layers = list()
    embedding_layers = list()

    # net categorical deep feature
    for col, num in categorical_columns_item.items():
        input_deep_cat_layer = Input(shape=(1,), name=col + "_categorical_deep_input")
        embedding_layer = Embedding(
            input_dim=num,
            output_dim=min(10, num // 2),
            embeddings_initializer=random_uniform(minval=-1, maxval=1),
            input_length=1,
            name=col + "_deep_embedding")(input_deep_cat_layer)
        embedding_layer = (
            Reshape(target_shape=(min(10, num // 2),), name=col + "_deep_reshape")(embedding_layer))
        embedding_layer = Dropout(rate=0.15, noise_shape=(None, 1), name=col + "_deep_dropout")(embedding_layer)
        input_layers.append(input_deep_cat_layer)
        embedding_layers.append(embedding_layer)

    # net numeric deep feature
    input_deep_num_layer = Input(
        shape=(num_deep_numeric_feature, ),
        name="numeric_deep_input")
    input_layers.append(input_deep_num_layer)

    # net numeric wide feature
    input_wide_num_layer = Input(
        shape=(num_wide_numeric_feature, ),
        name="numeric_wide_input")
    input_layers.append(input_wide_num_layer)

    hidden_layer = Dense(
        units=32,
        kernel_initializer=lecun_normal(),
        activation="selu")(
        Concatenate()([Concatenate()(embedding_layers), Dropout(rate=0.15)(input_deep_num_layer)]))
    hidden_layer = Dense(
        units=16,
        kernel_initializer=lecun_normal(),
        activation="selu")(hidden_layer)
    hidden_layer = Dense(
        units=8,
        kernel_initializer=lecun_normal(),
        activation="selu")(hidden_layer)
    hidden_layer = Concatenate()([hidden_layer, input_wide_num_layer])
    output_layer = Dense(
        units=1,
        kernel_initializer=lecun_normal(),
        bias_initializer=constant(logit(bias)),
        activation="sigmoid", name="output_layer")(hidden_layer)

    return Model(input_layers, output_layer)


def network_preformance(n_fold, net, trn_feature, val_feature, trn_label, val_label):
    pred_trn = net.predict(trn_feature).reshape((-1,))
    trn_gini = gini_normalized(trn_label.values.reshape((-1,)), pred_trn)

    pred_val = net.predict(val_feature).reshape((-1,))
    val_gini = gini_normalized(val_label.values.reshape((-1,)), pred_val)

    print("Fold %i prediction trn gini: %.5f" % (n_fold, trn_gini))
    print("Fold %i prediction val gini: %.5f" % (n_fold, val_gini))