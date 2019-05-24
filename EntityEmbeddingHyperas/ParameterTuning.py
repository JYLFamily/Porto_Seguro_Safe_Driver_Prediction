# coding:utf-8

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import set_random_seed
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Dropout, Dense
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
set_random_seed(15)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


def data():
    train_feature_for_model = pickle.load(open(os.path.join("E:\\Kaggle\\Porto Seguro Safe Driver Prediction", "train_feature_for_model.pkl"), mode="rb"))
    train_label = pickle.load(open(os.path.join("E:\\Kaggle\\Porto Seguro Safe Driver Prediction", "train_label.pkl"), mode="rb"))
    validation_feature_for_model = pickle.load(open(os.path.join("E:\\Kaggle\\Porto Seguro Safe Driver Prediction", "validation_feature_for_model.pkl"), mode="rb"))
    validation_label = pickle.load(open(os.path.join("E:\\Kaggle\\Porto Seguro Safe Driver Prediction", "validation_label.pkl"), mode="rb"))

    return train_feature_for_model, train_label, validation_feature_for_model, validation_label


def create_model(train_feature_for_model, train_label, validation_feature_for_model, validation_label):
    numeric_columns = pickle.load(open(os.path.join("E:\\Kaggle\\Porto Seguro Safe Driver Prediction", "numeric_columns.pkl"), mode="rb"))
    categorical_columns_num_unique = pickle.load(open(os.path.join("E:\\Kaggle\\Porto Seguro Safe Driver Prediction", "categorical_columns_num_unique.pkl"), mode="rb"))

    input_layers = list()
    embedding_layers = list()

    # categorical feature
    for col, num in categorical_columns_num_unique.items():
        input_layer = Input(shape=(1,), name=col + "_input")
        # nun+1 的原因在于, 只有一个 level 的时候 1 // 2 = 0, 预处理应该删掉这个特征
        embedding_layer = Embedding(input_dim=num, output_dim=min(50, num // 2), input_length=1,
                                    name=col + "_embedding")(input_layer)
        embedding_layer = Reshape(target_shape=(min(50, num // 2),))(embedding_layer)
        input_layers.append(input_layer)
        embedding_layers.append(embedding_layer)

    # numeric feature
    input_layer = Input(shape=(len(numeric_columns),))
    input_layers.append(input_layer)

    hidden_layer_categorical = Dropout({{uniform(0, 0.1)}})(Concatenate()(embedding_layers))
    hidden_layer_numeric = Dropout({{uniform(0, 0.1)}})(input_layer)
    hidden_layer_categorical = Dense({{choice([32, 48, 64])}}, activation="relu")(hidden_layer_categorical)
    hidden_layer_numeric = Dense({{choice([16, 32, 48])}}, activation="relu")(hidden_layer_numeric)
    hidden_layer = Concatenate()([hidden_layer_categorical, hidden_layer_numeric])

    hidden_layer = Dense({{choice([16, 32, 48])}}, activation="relu")(hidden_layer)
    hidden_layer = Dropout(0.35)(hidden_layer)
    hidden_layer = Dense({{choice([16, 32, 48])}}, activation="relu")(hidden_layer)
    hidden_layer = Dropout(0.15)(hidden_layer)
    hidden_layer = Dense(10, activation="relu")(hidden_layer)
    hidden_layer = Dropout(0.15)(hidden_layer)
    output_layer = Dense(1, activation="sigmoid")(hidden_layer)

    net = Model(
        input_layers,
        output_layer
    )
    net.compile(loss="binary_crossentropy", optimizer={{choice(["rmsprop", "adam", "sgd"])}})
    history = net.fit(
        train_feature_for_model, train_label,
        batch_size=4096,
        epochs=15,
        verbose=2,
        validation_data=(validation_feature_for_model, validation_label)
    )

    return {"loss": np.min(history.history["val_loss"]), "status": STATUS_OK, "model": net}


if __name__ == "__main__":
    best_run, _ = optim.minimize(
        model=create_model,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )
    print("*" * 72)
    print(best_run)
    print("*" * 72)
