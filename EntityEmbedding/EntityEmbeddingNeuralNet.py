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
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit, logit
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
from EntityEmbedding.EntityEmbeddingTree import EntityEmbeddingTree
from EntityEmbedding.NeuralNetUtil import network
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
        self.__categorical_columns_item = dict()  # each fold clear

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

        # data clean
        self.__train_feature.rename(columns={"ps_car_08_cat": "ps_car_08_bin"}, inplace=True)  # ps_car_08_cat binary
        self.__test_feature.rename(columns={"ps_car_08_cat": "ps_car_08_bin"}, inplace=True)

        self.__train_feature = \
            self.__train_feature[[col for col in self.__train_feature.columns if not col.startswith("ps_calc_")]]
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        self.__numeric_columns = [col for col in self.__train_feature.columns if not col.endswith(("_bin", "_cat"))]
        self.__categorical_columns = [col for col in self.__train_feature.columns if col.endswith(("_bin", "_cat"))]

        # deep feature
        self.__train_deep_feature = self.__train_feature.copy(deep=True)
        self.__test_deep_feature = self.__test_feature.copy(deep=True)

        # wide feature
        eet = EntityEmbeddingTree(
            numeric_columns=self.__numeric_columns,
            categorical_columns=self.__categorical_columns
        )
        eet.fit(self.__train_feature, self.__train_label)

        encoder = BinaryEncoder()  # binary encoder need pandas dataframe input type str
        self.__train_wide_feature = encoder.fit_transform(eet.transform(self.__train_feature))
        self.__test_wide_feature = encoder.transform(eet.transform(self.__test_feature))
        del self.__train_feature, self.__test_feature
        gc.collect()

    def model_fit_predict(self):
        # blending
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        self.__sub_preds = np.zeros(shape=(self.__test_deep_feature.shape[0], ))

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
                X=self.__train_deep_feature, y=self.__train_label)):

            trn_deep_x, trn_y = \
                self.__train_deep_feature.iloc[trn_idx].copy(deep=True), self.__train_label.iloc[trn_idx].copy(deep=True)
            val_deep_x, val_y = \
                self.__train_deep_feature.iloc[val_idx].copy(deep=True), self.__train_label.iloc[val_idx].copy(deep=True)
            tes_deep_x = self.__test_deep_feature.copy(deep=True)

            trn_wide_x = self.__train_wide_feature.iloc[trn_idx].copy(deep=True)
            val_wide_x = self.__train_wide_feature.iloc[val_idx].copy(deep=True)
            tes_wide_x = self.__test_wide_feature.copy(deep=True)

            # deep feature categorical
            for col in tqdm(self.__categorical_columns):
                num_unique = trn_deep_x[col].nunique()

                if num_unique == 1:
                    trn_deep_x = trn_deep_x.drop([col], axis=1)
                    val_deep_x = val_deep_x.drop([col], axis=1)
                    tes_deep_x = tes_deep_x.drop([col], axis=1)
                else:

                    if trn_deep_x[col].isna().sum():  # train exist np.nan
                        trn_deep_x[col] = trn_deep_x[col].fillna("missing")

                        mode = trn_deep_x[col].value_counts().index[0]
                        categories = trn_deep_x[col].unique()

                        val_deep_x[col] = val_deep_x[col].fillna("missing")
                        val_deep_x[col] = val_deep_x[col].apply(lambda x: x if x in categories else mode)
                        tes_deep_x[col] = tes_deep_x[col].fillna("missing")
                        tes_deep_x[col] = tes_deep_x[col].apply(lambda x: x if x in categories else mode)

                    else:  # train not exist np.nan
                        mode = trn_deep_x[col].value_counts().index[0]
                        categories = trn_deep_x[col].unique()

                        val_deep_x[col] = val_deep_x[col].fillna(mode)
                        val_deep_x[col] = val_deep_x[col].apply(lambda x: x if x in categories else mode)
                        tes_deep_x[col] = tes_deep_x[col].fillna(mode)
                        tes_deep_x[col] = tes_deep_x[col].apply(lambda x: x if x in categories else mode)

                    trn_deep_x[col] = trn_deep_x[col].astype(str)
                    val_deep_x[col] = val_deep_x[col].astype(str)
                    tes_deep_x[col] = tes_deep_x[col].astype(str)

                    encoder = LabelEncoder()
                    encoder.fit(trn_deep_x[col])
                    trn_deep_x[col] = encoder.transform(trn_deep_x[col])
                    val_deep_x[col] = encoder.transform(val_deep_x[col])
                    tes_deep_x[col] = encoder.transform(tes_deep_x[col])

                    self.__categorical_columns_item[col] = len(encoder.classes_)

            # deep feature numeric
            scaler = StandardScaler()  # calc std, mean skip np.nan
            scaler.fit(trn_deep_x[self.__numeric_columns])
            trn_deep_x[self.__numeric_columns] = scaler.transform(trn_deep_x[self.__numeric_columns])
            val_deep_x[self.__numeric_columns] = scaler.transform(val_deep_x[self.__numeric_columns])
            tes_deep_x[self.__numeric_columns] = scaler.transform(tes_deep_x[self.__numeric_columns])

            trn_deep_x[self.__numeric_columns] = trn_deep_x[self.__numeric_columns].fillna(0.)
            val_deep_x[self.__numeric_columns] = val_deep_x[self.__numeric_columns].fillna(0.)
            tes_deep_x[self.__numeric_columns] = tes_deep_x[self.__numeric_columns].fillna(0.)

            trn_feature_for_model = []
            val_feature_for_model = []
            tes_feature_for_model = []

            for col in self.__categorical_columns_item.keys():
                trn_feature_for_model.append(trn_deep_x[col].values)
                val_feature_for_model.append(val_deep_x[col].values)
                tes_feature_for_model.append(tes_deep_x[col].values)

            trn_feature_for_model.append(trn_deep_x[self.__numeric_columns].values)
            val_feature_for_model.append(val_deep_x[self.__numeric_columns].values)
            tes_feature_for_model.append(tes_deep_x[self.__numeric_columns].values)

            trn_feature_for_model.append(trn_wide_x.values)
            val_feature_for_model.append(val_wide_x.values)
            tes_feature_for_model.append(tes_wide_x.values)

            self.__net = network(
                categorical_columns_item=self.__categorical_columns_item,
                num_deep_numeric_feature=len(trn_deep_x.columns) - len(self.__categorical_columns_item),
                num_wide_numeric_feature=len(trn_wide_x.columns),
                bias=trn_y.mean()
            )
            self.__net.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=[roc_auc_score])

            self.__net.fit(
                trn_feature_for_model,
                trn_y.values,
                epochs=35,
                batch_size=32,
                verbose=0,
                callbacks=[
                    TensorBoard(),
                    EarlyStopping(patience=5, restore_best_weights=True)
                ],
                validation_data=(val_feature_for_model, val_y.values)
            )

            pred_val = self.__net.predict(val_feature_for_model).reshape((-1,))
            cv_gini = gini_normalized(val_y.values.reshape((-1,)), pred_val)
            print("Fold %i prediction cv gini: %.5f" % (n_fold, cv_gini))

            pred_test = self.__net.predict(tes_feature_for_model).reshape((-1,))  # 2D shape -> 1D shape
            self.__sub_preds += logit(pred_test) / self.__folds.n_splits

            self.__categorical_columns_item.clear()
            del trn_wide_x, val_wide_x, tes_wide_x, trn_deep_x, val_deep_x, tes_deep_x
            gc.collect()

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




