# coding:utf-8

import numpy as np
from keras.models import Model
from scipy.special import logit
from tensorflow import set_random_seed
from keras.initializers import truncated_normal, lecun_normal, constant
from keras.layers import Input, Embedding, Reshape, Dropout, Concatenate, Dense
np.random.seed(7)
set_random_seed(7)


def gini(y_true, y_pred):
    num = len(y_true)
    a_c = y_true[np.argsort(y_pred)].cumsum()

    return (a_c.sum() / a_c[-1] - (num + 1) / 2.0) / num


def gini_normalized(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)


def network(col_num_categorical_feature, num_numeric_feature, bias):
    input_layers = list()
    embedding_layers = list()

    # categorical feature
    for col, num in col_num_categorical_feature.items():
        input_cat_layer = Input(shape=(1,), name=col + "_cat_input")
        embedding_layer = Embedding(
            input_dim=num,
            output_dim=min(10, num // 2),
            embeddings_initializer=truncated_normal(mean=0, stddev=1/np.sqrt(num)),
            input_length=1,
            name=col + "_cat_embedding")(input_cat_layer)
        embedding_layer = (
            Reshape(target_shape=(min(10, num // 2),), name=col + "_cat_reshape")(embedding_layer))
        embedding_layer = Dropout(rate=0.15, noise_shape=(None, 1), name=col + "_cat_dropout")(embedding_layer)
        input_layers.append(input_cat_layer)
        embedding_layers.append(embedding_layer)

    # numeric feature
    input_num_layer = Input(
        shape=(num_numeric_feature, ),
        name="num_input")
    input_layers.append(input_num_layer)

    hidden_layer = Dense(
        units=32,
        kernel_initializer=lecun_normal(),
        activation="selu",
        name="dense_1")(
        Concatenate()([Concatenate()(embedding_layers), Dropout(rate=0.15, name="num_dropout")(input_num_layer)]))
    hidden_layer = Dense(
        units=16,
        kernel_initializer=lecun_normal(),
        activation="selu",
        name="dense_2")(hidden_layer)
    hidden_layer = Dense(
        units=8,
        kernel_initializer=lecun_normal(),
        activation="selu",
        name="dense_3")(hidden_layer)
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