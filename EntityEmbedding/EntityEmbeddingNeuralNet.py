# coding:utf-8

import os
import gc
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as k
from tqdm import tqdm
from tensorflow import set_random_seed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import BinaryEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Dense
from keras.initializers import random_uniform, lecun_normal, constant
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit, logit
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
from EntityEmbedding.EntityEmbeddingTree import EntityEmbeddingTree
np.random.seed(7)
set_random_seed(7)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


def gini(y_true, y_pred):
    num = len(y_true)
    a_c = y_true[np.argsort(y_pred)].cumsum()

    return (a_c.sum() / a_c[-1] - (num + 1) / 2.0) / num


def gini_normalized(y_true, y_pred):
    return gini(y_true, y_pred) / gini(y_true, y_true)


def roc_auc_score(y_true, y_pred):
    score = tf.metrics.auc(y_true, y_pred)[1]
    k.get_session().run(tf.local_variables_initializer())
    return score


class EntityEmbeddingNeuralNet(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_index = [None for _ in range(2)]

        self.__train_wide_feature, self.__test_wide_feature = [pd.DataFrame() for _ in range(2)]
        self.__train_deep_feature, self.__test_deep_feature = [pd.DataFrame() for _ in range(2)]

        self.__numeric_columns = list()
        self.__categorical_columns = list()
        self.__categorical_columns_item = dict()

        # model fit predict
        self.__folds = None
        self.__sub_preds = None

        self.__net = None
        self.__early_stopping = None

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"), na_values=-1)
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"), na_values=-1)

    def data_prepare(self):
        self.__train_feature, self.__train_label = self.__train.iloc[:, 2:].copy(), self.__train.iloc[:, 1].copy()
        self.__test_feature, self.__test_index = self.__test.iloc[:, 1:].copy(), self.__test.iloc[:, [0]].copy()
        del self.__train, self.__test
        gc.collect()

        # delete ps_calc feature
        self.__train_feature = \
            self.__train_feature[[col for col in self.__train_feature.columns if not col.startswith("ps_calc_")]]
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        # endwith _bin or _cat categorical
        self.__numeric_columns = [col for col in self.__train_feature.columns if not col.endswith(("_bin", "_cat"))]
        self.__categorical_columns = [col for col in self.__train_feature.columns if col.endswith(("_bin", "_cat"))]

        # wide feature
        eet = EntityEmbeddingTree(
            numeric_columns=self.__numeric_columns,
            categorical_columns=self.__categorical_columns
        )
        eet.fit(self.__train_feature, self.__train_label)

        # binary encoder need pandas dataframe input type str
        encoder = BinaryEncoder()
        self.__train_wide_feature = encoder.fit_transform(eet.transform(self.__train_feature))
        self.__test_wide_feature = encoder.transform(eet.transform(self.__test_feature))

        # deep feature categorical
        for col in tqdm(self.__categorical_columns):
            num_unique = self.__train_feature[col].nunique()

            if num_unique == 1:
                self.__train_feature = self.__train_feature.drop([col], axis=1)
                self.__test_feature = self.__test_feature.drop([col], axis=1)
            else:

                if self.__train_feature[col].isna().sum():  # train 存在缺失
                    self.__train_feature[col] = self.__train_feature[col].fillna("missing")
                    mode = self.__train_feature[col].value_counts().index[0]
                    categories = self.__train_feature[col].unique()

                    self.__test_feature[col] = self.__test_feature[col].fillna("missing")
                    self.__test_feature[col] = [mode if i not in categories else i for i in self.__test_feature[col]]

                else:  # train 不存在缺失
                    mode = self.__train_feature[col].value_counts().index[0]
                    categories = self.__train_feature[col].unique()
                    self.__test_feature[col] = self.__test_feature[col].fillna(mode)
                    self.__test_feature[col] = [mode if i not in categories else i for i in self.__test_feature[col]]

                self.__train_feature[col] = self.__train_feature[col].astype(str)
                self.__test_feature[col] = self.__test_feature[col].astype(str)

                encoder = LabelEncoder()
                encoder.fit(self.__train_feature[col])
                self.__train_feature[col] = encoder.transform(self.__train_feature[col])
                self.__test_feature[col] = encoder.transform(self.__test_feature[col])
                self.__categorical_columns_item[col] = len(encoder.classes_)
        self.__categorical_columns = self.__categorical_columns_item.keys()

        # deep feature numeric
        scaler = StandardScaler()  # calc std, mean skip np.nan
        scaler.fit(self.__train_feature[self.__numeric_columns])
        self.__train_feature[self.__numeric_columns] = scaler.transform(self.__train_feature[self.__numeric_columns])
        self.__test_feature[self.__numeric_columns] = scaler.transform(self.__test_feature[self.__numeric_columns])

        self.__train_feature[self.__numeric_columns] = self.__train_feature[self.__numeric_columns].fillna(0.0)
        self.__test_feature[self.__numeric_columns] = self.__test_feature[self.__numeric_columns].fillna(0.0)

        self.__train_deep_feature, self.__test_deep_feature = self.__train_feature, self.__test_feature
        del self.__train_feature, self.__test_feature
        gc.collect()

    def model_fit_predict(self):
        # blending
        # self.__folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        # self.__sub_preds = np.zeros(shape=(self.__test_wide_feature.shape[0], ))
        #
        # for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
        #         X=self.__train_wide_feature, y=self.__train_label)):
        #
        #     trn_wide_x = self.__train_wide_feature.iloc[trn_idx]
        #     val_wide_x = self.__train_wide_feature.iloc[val_idx]
        #
        #     trn_deep_x, trn_y = self.__train_deep_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
        #     val_deep_x, val_y = self.__train_deep_feature.iloc[val_idx], self.__train_label.iloc[val_idx]
        #
        #     tra_feature_for_model = []
        #     val_feature_for_model = []
        #     test_feature_for_model = []
        #
        #     for col in self.__categorical_columns:
        #         tra_feature_for_model.append(trn_deep_x[col].values)
        #         val_feature_for_model.append(val_deep_x[col].values)
        #         test_feature_for_model.append(self.__test_deep_feature[col].values)
        #
        #     tra_feature_for_model.append(trn_deep_x[self.__numeric_columns].values)
        #     val_feature_for_model.append(val_deep_x[self.__numeric_columns].values)
        #     test_feature_for_model.append(self.__test_deep_feature[self.__numeric_columns].values)
        #
        #     tra_feature_for_model.append(trn_wide_x.values)
        #     val_feature_for_model.append(val_wide_x.values)
        #     test_feature_for_model.append(self.__test_wide_feature.values)
        #
        #     del trn_wide_x, val_wide_x, trn_deep_x, val_deep_x
        #
        #     # net
        #     input_layers = list()
        #     embedding_layers = list()
        #
        #     # net categorical deep feature
        #     for col, num in self.__categorical_columns_item.items():
        #         input_deep_cat_layer = Input(shape=(1,), name=col + "_categorical_deep_input")
        #         embedding_layer = Embedding(
        #             input_dim=num,
        #             output_dim=min(50, num // 2),
        #             embeddings_initializer=random_uniform(minval=-1, maxval=1),
        #             input_length=1,
        #             name=col + "_deep_embedding")(input_deep_cat_layer)
        #         embedding_layer = (
        #             Reshape(target_shape=(min(50, num // 2), ), name=col + "_deep_reshape")(embedding_layer))
        #         input_layers.append(input_deep_cat_layer)
        #         embedding_layers.append(embedding_layer)
        #
        #     # net numeric deep feature
        #     input_deep_num_layer = Input(
        #         shape=(len(self.__train_deep_feature.columns) - len(self.__categorical_columns_item),),
        #         name="numeric_deep_input")
        #     input_layers.append(input_deep_num_layer)
        #
        #     # net numeric wide feature
        #     input_wide_layer = Input(shape=(self.__train_wide_feature.shape[1],), name="numeric_wide_input")
        #     input_layers.append(input_wide_layer)
        #
        #     hidden_layer = Dense(
        #         units=48,
        #         kernel_initializer=lecun_normal(),
        #         activation="selu")(
        #         Concatenate()([Concatenate()(embedding_layers), input_deep_num_layer]))
        #     hidden_layer = Dense(
        #         units=24,
        #         kernel_initializer=lecun_normal(),
        #         activation="selu")(hidden_layer)
        #     hidden_layer = Dense(
        #         units=12,
        #         kernel_initializer=lecun_normal(),
        #         activation="selu")(hidden_layer)
        #     hidden_layer = Concatenate()([hidden_layer, input_wide_layer])
        #     output_layer = Dense(
        #         units=1,
        #         kernel_initializer=lecun_normal(),
        #         bias_initializer=constant(logit(trn_y.mean())),
        #         activation="sigmoid", name="output_layer")(hidden_layer)
        #
        #     self.__net = Model(input_layers, output_layer)
        #     self.__net.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=[roc_auc_score])
        #
        #     self.__net.fit(
        #         tra_feature_for_model,
        #         trn_y.values,
        #         epochs=35,
        #         batch_size=32,
        #         verbose=0,
        #         callbacks=[
        #             TensorBoard(),
        #             EarlyStopping(patience=5, restore_best_weights=True)
        #         ],
        #         validation_data=(val_feature_for_model, val_y.values)
        #     )
        #
        #     pred_val = self.__net.predict(val_feature_for_model).reshape((-1,))
        #     cv_gini = gini_normalized(val_y.values.reshape((-1,)), pred_val)
        #     print("Fold %i prediction cv gini: %.5f" % (n_fold, cv_gini))
        #
        #     pred_test = self.__net.predict(test_feature_for_model).reshape((-1,))  # 2D shape -> 1D shape
        #     self.__sub_preds += logit(pred_test) / self.__folds.n_splits

        train_feature_for_model = []
        test_feature_for_model = []

        for col in self.__categorical_columns:
            train_feature_for_model.append(self.__train_deep_feature[col].values)
            test_feature_for_model.append(self.__test_deep_feature[col].values)

        train_feature_for_model.append(self.__train_deep_feature[self.__numeric_columns].values)
        test_feature_for_model.append(self.__test_deep_feature[self.__numeric_columns].values)

        train_feature_for_model.append(self.__train_wide_feature.values)
        test_feature_for_model.append(self.__test_wide_feature.values)

        # net
        input_layers = list()
        embedding_layers = list()
        # net categorical deep feature
        for col, num in self.__categorical_columns_item.items():
            input_deep_cat_layer = Input(shape=(1,), name=col + "_categorical_deep_input")
            embedding_layer = Embedding(
                input_dim=num,
                output_dim=min(50, num // 2),
                input_length=1,
                name=col + "_deep_embedding")(input_deep_cat_layer)
            embedding_layer = Reshape(target_shape=(min(50, num // 2), ), name=col + "_deep_reshape")(embedding_layer)
            input_layers.append(input_deep_cat_layer)
            embedding_layers.append(embedding_layer)
        # net numeric deep feature
        input_deep_num_layer = Input(
            shape=(len(self.__train_deep_feature.columns) - len(self.__categorical_columns_item),),
            name="numeric_deep_input")
        input_layers.append(input_deep_num_layer)
        # net numeric wide feature
        input_wide_layer = Input(shape=(self.__train_wide_feature.shape[1],), name="numeric_wide_input")
        input_layers.append(input_wide_layer)

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
        hidden_layer = Concatenate()([hidden_layer, input_wide_layer])
        output_layer = Dense(
            units=1,
            kernel_initializer=lecun_normal(),
            bias_initializer=constant(logit(self.__train_label.mean())),
            activation="sigmoid",
            name="output_layer")(hidden_layer)
        self.__net = Model(input_layers, output_layer)
        self.__net.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=[roc_auc_score])

        self.__net.fit(
                 train_feature_for_model,
                 self.__train_label.values,
                 epochs=25,
                 batch_size=32,
                 verbose=2
            )

        self.__sub_preds = self.__net.predict(test_feature_for_model)

    def data_write(self):
        self.__test_index["target"] = expit(self.__sub_preds.reshape((-1,)))
        self.__test_index.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file)

    eenn = EntityEmbeddingNeuralNet(input_path=config["input_path"], output_path=config["output_path"])
    eenn.data_read()
    eenn.data_prepare()
    eenn.model_fit_predict()
    eenn.data_write()




