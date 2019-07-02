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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit, logit
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from EntityEmbedding.NeuralNetUtil import network, network_preformance
np.random.seed(7)
set_random_seed(7)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


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
        self.__train_feature, self.__train_label = (
            self.__train.iloc[:, 2:].copy(deep=True), self.__train.iloc[:, 1].copy(deep=True))
        self.__test_feature, self.__test_index = (
            self.__test.iloc[:, 1:].copy(deep=True), self.__test.iloc[:, [0]].copy(deep=True))
        del self.__train, self.__test
        gc.collect()

        # data clean
        self.__train_feature.rename(columns={
            "ps_ind_01": "ps_ind_01_num_cat",   # nunique 8
            "ps_ind_03": "ps_ind_03_num_cat",   # nunique 12
            "ps_ind_14": "ps_ind_14_num_cat",   # nunique 5
            "ps_ind_15": "ps_ind_15_num_cat",   # nunique 14
            "ps_reg_01": "ps_reg_01_num_cat",   # nunique 10
            "ps_reg_02": "ps_reg_02_num_cat",   # nunqiue 19
            "ps_car_11": "ps_car_11_num_cat",   # nunique 5
            "ps_car_15": "ps_car_15_num_cat"},  # nunique 15
            inplace=True
        )
        self.__test_feature.rename(columns={
            "ps_ind_01": "ps_ind_01_num_cat",
            "ps_ind_03": "ps_ind_03_num_cat",
            "ps_ind_14": "ps_ind_14_num_cat",
            "ps_ind_15": "ps_ind_15_num_cat",
            "ps_reg_01": "ps_reg_01_num_cat",
            "ps_reg_02": "ps_reg_02_num_cat",
            "ps_car_11": "ps_car_11_num_cat",
            "ps_car_15": "ps_car_15_num_cat"},
            inplace=True
        )

        self.__train_feature = (
            self.__train_feature[[col for col in self.__train_feature.columns if not col.startswith("ps_calc_")]])
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        self.__numeric_columns = [col for col in self.__train_feature.columns if not col.endswith(("_bin", "_cat"))]
        self.__categorical_columns = [col for col in self.__train_feature.columns if col.endswith(("_bin", "_cat"))]

        # deep feature

        # wide feature
        # from EntityEmbedding.EntityEmbeddingTree import EntityEmbeddingTree
        # eet = EntityEmbeddingTree(
        #     numeric_columns=self.__numeric_columns,
        #     categorical_columns=self.__categorical_columns
        # )
        # eet.fit(self.__train_feature, self.__train_label)
        #
        # from category_encoders import BinaryEncoder
        # encoder = BinaryEncoder()  # binary encoder need pandas dataframe input type str
        # self.__train_wide_feature = encoder.fit_transform(eet.transform(self.__train_feature))
        # self.__test_wide_feature = encoder.transform(eet.transform(self.__test_feature))
        # del self.__train_feature, self.__test_feature
        # gc.collect()

    def model_fit_predict(self):
        # blending
        self.__folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        self.__sub_preds = np.zeros(shape=(self.__test_feature.shape[0], ))

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(
                X=self.__train_feature, y=self.__train_label)):

            trn_x = self.__train_feature.iloc[trn_idx].copy(deep=True)
            val_x = self.__train_feature.iloc[val_idx].copy(deep=True)
            tes_x = self.__test_feature.copy(deep=True)

            trn_y = self.__train_label.iloc[trn_idx].copy(deep=True)
            val_y = self.__train_label.iloc[val_idx].copy(deep=True)

            # deep feature categorical
            for col in tqdm(self.__categorical_columns):
                num_unique = trn_x[col].nunique()

                if num_unique == 1:
                    trn_x = trn_x.drop([col], axis=1)
                    val_x = val_x.drop([col], axis=1)
                    tes_x = tes_x.drop([col], axis=1)
                else:

                    if trn_x[col].isna().sum():  # train exist np.nan
                        trn_x[col] = trn_x[col].fillna("missing")

                        mode = trn_x[col].value_counts().index[0]
                        categories = trn_x[col].unique()

                        val_x[col] = val_x[col].fillna("missing")
                        val_x[col] = val_x[col].apply(lambda x: x if x in categories else mode)
                        tes_x[col] = tes_x[col].fillna("missing")
                        tes_x[col] = tes_x[col].apply(lambda x: x if x in categories else mode)

                    else:  # train not exist np.nan
                        mode = trn_x[col].value_counts().index[0]
                        categories = trn_x[col].unique()

                        val_x[col] = val_x[col].fillna(mode)
                        val_x[col] = val_x[col].apply(lambda x: x if x in categories else mode)
                        tes_x[col] = tes_x[col].fillna(mode)
                        tes_x[col] = tes_x[col].apply(lambda x: x if x in categories else mode)

                    trn_x[col] = trn_x[col].astype(str)
                    val_x[col] = val_x[col].astype(str)
                    tes_x[col] = tes_x[col].astype(str)

                    encoder = LabelEncoder()
                    encoder.fit(trn_x[col])
                    trn_x[col] = encoder.transform(trn_x[col])
                    val_x[col] = encoder.transform(val_x[col])
                    tes_x[col] = encoder.transform(tes_x[col])

                    self.__categorical_columns_item[col] = len(encoder.classes_)

            # deep feature
            trn_deep_x = trn_x.copy(deep=True)
            val_deep_x = val_x.copy(deep=True)
            tes_deep_x = tes_x.copy(deep=True)

            scaler = StandardScaler()  # calc std, mean skip np.nan
            scaler.fit(trn_deep_x[self.__numeric_columns])
            trn_deep_x[self.__numeric_columns] = scaler.transform(trn_deep_x[self.__numeric_columns])
            val_deep_x[self.__numeric_columns] = scaler.transform(val_deep_x[self.__numeric_columns])
            tes_deep_x[self.__numeric_columns] = scaler.transform(tes_deep_x[self.__numeric_columns])

            trn_deep_x[self.__numeric_columns] = trn_deep_x[self.__numeric_columns].fillna(0.)
            val_deep_x[self.__numeric_columns] = val_deep_x[self.__numeric_columns].fillna(0.)
            tes_deep_x[self.__numeric_columns] = tes_deep_x[self.__numeric_columns].fillna(0.)

            # wide feature
            wide_feature = [col for col in self.__categorical_columns_item.keys() if not col.endswith("_num_cat")]
            trn_wide_x = trn_x[wide_feature].copy(deep=True)
            val_wide_x = val_x[wide_feature].copy(deep=True)
            tes_wide_x = tes_x[wide_feature].copy(deep=True)

            encoder = OneHotEncoder(categories="auto", sparse=False)
            encoder.fit(trn_wide_x)
            trn_wide_x = encoder.transform(trn_wide_x)
            val_wide_x = encoder.transform(val_wide_x)
            tes_wide_x = encoder.transform(tes_wide_x)

            del trn_x, val_x, tes_x
            gc.collect()

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

            # trn_feature_for_model.append(trn_wide_x.values)
            # val_feature_for_model.append(val_wide_x.values)
            # tes_feature_for_model.append(tes_wide_x.values)

            trn_feature_for_model.append(trn_wide_x)
            val_feature_for_model.append(val_wide_x)
            tes_feature_for_model.append(tes_wide_x)

            self.__net = network(
                categorical_columns_item=self.__categorical_columns_item,
                num_deep_numeric_feature=len(self.__numeric_columns),
                num_wide_numeric_feature=trn_wide_x.shape[1],
                bias=trn_y.mean()
            )
            self.__net.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=[roc_auc_score])

            self.__net.fit(
                x=trn_feature_for_model,
                y=trn_y.values,
                epochs=35,
                batch_size=256,
                verbose=2,
                callbacks=[
                    EarlyStopping(
                        patience=5,
                        restore_best_weights=True
                    )],
                validation_data=(val_feature_for_model, val_y.values)
            )

            network_preformance(
                n_fold=n_fold,
                net=self.__net,
                trn_feature=trn_feature_for_model,
                val_feature=val_feature_for_model,
                trn_label=trn_y,
                val_label=val_y
            )

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




