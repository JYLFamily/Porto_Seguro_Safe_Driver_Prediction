# coding:utf-8

import os
import numpy as np
import pandas as pd
from tensorflow import set_random_seed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
set_random_seed(15)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


class EntityEmbeddingNeuralNet(object):
    def __init__(self, *, input_path, output_path, log_path):
        self.__input_path, self.__output_path, self.__log_path = input_path, output_path, log_path

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_id = [None for _ in range(2)]

        self.__numeric_columns = list()
        self.__categorical_columns = list()
        self.__categorical_columns_num_unique = dict()  # categorical feature 的 columns: num_unique

        # self.__train_feature_for_model = []
        self.__test_feature_for_model = []

        # model fit predict
        self.__folds = None
        self.__sub_preds = None

        self.__net = None
        self.__tensor_board = None
        self.__early_stopping = None  # epoch 的 early stop
        self.__model_checkpoint = None  # train 的不同时间点, 保存模型

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = self.__train.iloc[:, 2:].copy(), self.__train.iloc[:, 1].copy()
        self.__test_feature, self.__test_id = self.__test.iloc[:, 1:].copy(), self.__test.iloc[:, [0]].copy()

        # 删除了所有以 ps_calc 开头的 feature
        self.__train_feature = self.__train_feature[[col for col in self.__train_feature.columns if not col.startswith("ps_calc_")]]
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        self.__numeric_columns = [col for col in self.__train_feature.columns if not col.endswith("_cat")]
        self.__categorical_columns = [col for col in self.__train_feature.columns if col.endswith("_cat")]

        # numeric
        for col in tqdm(self.__numeric_columns):
            self.__train_feature[col] = self.__train_feature[col].fillna(0)  # FIX BUG np.log1p 报错
            self.__test_feature[col] = self.__test_feature[col].fillna(0)

        scaler = StandardScaler()
        scaler.fit(self.__train_feature[self.__numeric_columns])
        self.__train_feature[self.__numeric_columns] = scaler.transform(self.__train_feature[self.__numeric_columns])
        self.__test_feature[self.__numeric_columns] = scaler.transform(self.__test_feature[self.__numeric_columns])

        # categorical
        for col in tqdm(self.__categorical_columns):
            num_unique = self.__train_feature[col].nunique()

            if num_unique == 1:
                self.__train_feature = self.__train_feature.drop([col], axis=1)
                self.__test_feature = self.__test_feature.drop([col], axis=1)
                self.__categorical_columns.remove(col)
            else:

                if self.__train_feature[col].isna().sum():  # train 存在缺失
                    self.__train_feature[col] = self.__train_feature[col].fillna("missing")
                    mode = self.__train_feature[col].value_counts().index[0]
                    categories = self.__train_feature[col].unique()

                    self.__test_feature[col] = self.__test_feature[col].fiillna(mode)
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
                self.__categorical_columns_num_unique[col] = len(encoder.classes_)

                # self.__train_feature_for_model.append(self.__train_feature[col].values)
                self.__test_feature_for_model.append(self.__test_feature[col].values)

        # self.__train_feature_for_model.append(self.__train_feature[self.__numeric_columns].values)
        self.__test_feature_for_model.append(self.__test_feature[self.__numeric_columns].values)

        del self.__train, self.__test

    def model_fit_predict(self):
        input_layers = list()
        embedding_layers = list()

        # categorical feature
        for col, num in self.__categorical_columns_num_unique.items():
            input_layer = Input(shape=(1, ), name=col+"_input")
            # nun+1 的原因在于, 只有一个 level 的时候 1 // 2 = 0, 预处理应该删掉这个特征
            embedding_layer = Embedding(input_dim=num, output_dim=min(50, num // 2), input_length=1, name=col+"_embedding")(input_layer)
            embedding_layer = Reshape(target_shape=(min(50, num // 2), ))(embedding_layer)
            input_layers.append(input_layer)
            embedding_layers.append(embedding_layer)

        # numeric feature
        input_layer = Input(shape=(len(self.__train_feature.columns) - len(self.__categorical_columns_num_unique), ))
        input_layers.append(input_layer)

        hidden_layer_categorical = Dropout(0.02)(Concatenate()(embedding_layers))
        hidden_layer_numeric = Dropout(0.02)(input_layer)
        hidden_layer_categorical = Dense(40, activation="relu")(hidden_layer_categorical)
        hidden_layer_numeric = Dense(40, activation="relu")(hidden_layer_numeric)
        hidden_layer = Concatenate()([hidden_layer_categorical, hidden_layer_numeric])

        hidden_layer = Dense(40, activation="relu")(hidden_layer)
        hidden_layer = Dropout(.35)(hidden_layer)
        hidden_layer = Dense(20, activation="relu")(hidden_layer)
        hidden_layer = Dropout(.15)(hidden_layer)
        hidden_layer = Dense(10, activation="relu")(hidden_layer)
        hidden_layer = Dropout(.15)(hidden_layer)
        output_layer = Dense(1, activation="sigmoid")(hidden_layer)

        self.__net = Model(
            input_layers,
            output_layer
        )

        # stacking
        self.__folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        self.__sub_preds = np.zeros(shape=self.__test_feature.shape[0])

        for n_fold, (trn_idx, val_idx) in enumerate(self.__folds.split(X=self.__train_feature, y=self.__train_label)):
            trn_x, trn_y = self.__train_feature.iloc[trn_idx], self.__train_label.iloc[trn_idx]
            val_x, val_y = self.__train_feature.iloc[val_idx], self.__train_label.iloc[val_idx]

            tra_feature_for_model = []
            val_feature_for_model = []

            for col in self.__categorical_columns:
                tra_feature_for_model.append(trn_x[col])
                val_feature_for_model.append(val_x[col])

            tra_feature_for_model.append(trn_x[self.__numeric_columns].values)
            val_feature_for_model.append(val_x[self.__numeric_columns].values)

            self.__tensor_board = TensorBoard(
                log_dir=self.__log_path,
                histogram_freq=1  # 需要制定 validation_data
            )
            self.__early_stopping = EarlyStopping(patience=4, verbose=1)
            self.__model_checkpoint = ModelCheckpoint(filepath="net.h5", save_best_only=True)

            self.__net.compile(loss="binary_crossentropy", optimizer="adam")
            self.__net.fit(
                tra_feature_for_model,
                trn_y.values,
                epochs=20,
                batch_size=4096,
                validation_data=(val_feature_for_model, val_y.values),
                callbacks=[self.__early_stopping, self.__model_checkpoint, self.__tensor_board]
            )
            pred_test = self.__net.predict(self.__test_feature_for_model)  # 2D shape
            self.__sub_preds += pred_test.reshape((-1, )) / self.__folds.n_splits

        self.__test_id["target"] = self.__sub_preds

    def data_write(self):
        self.__test_id.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    eenn = EntityEmbeddingNeuralNet(
        input_path="E:\\Kaggle\\Porto Seguro Safe Driver Prediction",
        output_path="C:\\Users\\jiangyilan\\Desktop",
        log_path="C:\\Users\\jiangyilan\\PycharmProjects\\Porto_Seguro-s_Safe_Driver_Prediction\\logs"
    )
    eenn.data_read()
    eenn.data_prepare()
    eenn.model_fit_predict()
    eenn.data_write()




