# coding:utf-8

from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Dense
from keras.initializers import random_uniform, lecun_normal, constant
from scipy.special import logit
from tensorflow import set_random_seed
set_random_seed(7)


def network(categorical_columns_item, num_deep_numeric_feature, num_wide_numeric_feature, bias):
    input_layers = list()
    embedding_layers = list()

    # net categorical deep feature
    for col, num in categorical_columns_item.items():
        input_deep_cat_layer = Input(shape=(1,), name=col + "_categorical_deep_input")
        embedding_layer = Embedding(
            input_dim=num,
            output_dim=min(50, num // 2),
            embeddings_initializer=random_uniform(minval=-1, maxval=1),
            input_length=1,
            name=col + "_deep_embedding")(input_deep_cat_layer)
        embedding_layer = (
            Reshape(target_shape=(min(50, num // 2),), name=col + "_deep_reshape")(embedding_layer))
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
        units=48,
        kernel_initializer=lecun_normal(),
        activation="selu")(
        Concatenate()([Concatenate()(embedding_layers), input_deep_num_layer]))
    hidden_layer = Dense(
        units=24,
        kernel_initializer=lecun_normal(),
        activation="selu")(hidden_layer)
    hidden_layer = Dense(
        units=12,
        kernel_initializer=lecun_normal(),
        activation="selu")(hidden_layer)
    hidden_layer = Concatenate()([hidden_layer, input_wide_num_layer])
    output_layer = Dense(
        units=1,
        kernel_initializer=lecun_normal(),
        bias_initializer=constant(logit(bias)),
        activation="sigmoid", name="output_layer")(hidden_layer)

    return Model(input_layers, output_layer)