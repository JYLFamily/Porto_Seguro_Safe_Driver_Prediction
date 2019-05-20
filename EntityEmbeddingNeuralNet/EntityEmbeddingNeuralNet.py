# coding:utf-8

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow import set_random_seed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, Concatenate, Dense, BatchNormalization
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
np.random.seed(7)
set_random_seed(7)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


class EntityEmbeddingNeuralNet(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_id = [None for _ in range(2)]

        self.__numeric_columns = list()
        self.__categorical_columns = list()
        self.__categorical_columns_num_unique = dict()  # categorical feature 的 columns: num_unique

        self.__train_feature_for_model = []
        self.__test_feature_for_model = []

        # model fit predict
        self.__folds = None
        self.__sub_preds = None

        self.__net = None
        self.__early_stopping = None  # epoch 的 early stop

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"), na_values=-1)
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"), na_values=-1)

    def data_prepare(self):
        self.__train_feature, self.__train_label = self.__train.iloc[:, 2:].copy(), self.__train.iloc[:, 1].copy()
        self.__test_feature, self.__test_id = self.__test.iloc[:, 1:].copy(), self.__test.iloc[:, [0]].copy()

        # 删除了所有以 ps_calc 开头的 feature
        self.__train_feature = \
            self.__train_feature[[col for col in self.__train_feature.columns if not col.startswith("ps_calc_")]]
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        # _bin, _cat 结尾的特征为需要 Entity Embedding 的特征
        self.__numeric_columns = [col for col in self.__train_feature.columns if not col.endswith(("_bin", "_cat"))]
        self.__categorical_columns = [col for col in self.__train_feature.columns if col.endswith(("_bin", "_cat"))]

        # numeric
        scaler = StandardScaler()  # 计算 std, mean 时, 跳过 np.nan
        scaler.fit(self.__train_feature[self.__numeric_columns])
        self.__train_feature[self.__numeric_columns] = scaler.transform(self.__train_feature[self.__numeric_columns])
        self.__test_feature[self.__numeric_columns] = scaler.transform(self.__test_feature[self.__numeric_columns])
        self.__train_feature[self.__numeric_columns] = self.__train_feature[self.__numeric_columns].fillna(0.0)
        self.__test_feature[self.__numeric_columns] = self.__test_feature[self.__numeric_columns].fillna(0.0)

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
                self.__categorical_columns_num_unique[col] = len(encoder.classes_)

                self.__train_feature_for_model.append(self.__train_feature[col].values)
                self.__test_feature_for_model.append(self.__test_feature[col].values)

        self.__train_feature_for_model.append(self.__train_feature[self.__numeric_columns].values)
        self.__test_feature_for_model.append(self.__test_feature[self.__numeric_columns].values)

        del self.__train, self.__test

    def model_fit_predict(self):
        input_layers = list()
        embedding_layers = list()

        # categorical feature
        for col, num in self.__categorical_columns_num_unique.items():
            input_layer = Input(shape=(1, ), name=col + "_input")
            embedding_layer = Embedding(
                input_dim=num, output_dim=min(10, num // 2), input_length=1, name=col + "_embedding")(input_layer)
            embedding_layer = Reshape(target_shape=(min(10, num // 2), ), name=col + "_reshape")(embedding_layer)
            input_layers.append(input_layer)
            embedding_layers.append(embedding_layer)

        # numeric feature
        input_layer = Input(
            shape=(len(self.__train_feature.columns) - len(self.__categorical_columns_num_unique), ),
            name="numeric_input")
        input_layers.append(input_layer)

        hidden_layer = Dense(units=16, activation="relu")(Concatenate()([Concatenate()(embedding_layers), input_layer]))
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dense(units=8, activation="relu")(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Dense(units=4, activation="relu")(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        output_layer = Dense(1, activation="sigmoid", name="output_layer")(hidden_layer)

        self.__net = Model(input_layers, output_layer)
        plot_model(self.__net, to_file="net.png", show_shapes=True, show_layer_names=True)

        # stacking
        self.__folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        self.__sub_preds = np.zeros(shape=(self.__test_feature.shape[0], ))

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

            self.__early_stopping = EarlyStopping(patience=4, verbose=2)
            self.__net.compile(loss="binary_crossentropy", optimizer=Adam())

            history = self.__net.fit(
                tra_feature_for_model,
                trn_y.values,
                epochs=20,
                batch_size=32,
                validation_data=(val_feature_for_model, val_y.values),
                callbacks=[self.__early_stopping]
            )
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()

            pred_test = self.__net.predict(self.__test_feature_for_model)  # 2D shape
            self.__sub_preds += pred_test.reshape((-1, )) / self.__folds.n_splits

        self.__test_id["target"] = self.__sub_preds

    def data_write(self):
        self.__test_id.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)


if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file)

    eenn = EntityEmbeddingNeuralNet(input_path=config["input_path"], output_path=config["output_path"])
    eenn.data_read()
    eenn.data_prepare()
    eenn.model_fit_predict()
    eenn.data_write()




