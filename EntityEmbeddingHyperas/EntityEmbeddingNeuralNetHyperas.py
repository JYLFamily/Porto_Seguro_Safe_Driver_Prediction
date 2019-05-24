# coding:utf-8

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import set_random_seed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
set_random_seed(15)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


class EntityEmbeddingNeuralNetHyperas(object):
    def __init__(self, *, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # data prepare
        self.__train, self.__validation, self.__test = [None for _ in range(3)]
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__validation_feature, self.__validation_label = [None for _ in range(2)]
        self.__test_feature, self.__test_id = [None for _ in range(2)]

        self.__numeric_columns = list()
        self.__categorical_columns = list()
        self.__categorical_columns_num_unique = dict()  # categorical feature 的 columns: num_unique

        self.__train_feature_for_model = []
        self.__validation_feature_for_model = []
        self.__test_feature_for_model = []

        # data write

    def data_read(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))

    def data_prepare(self):
        self.__train_feature, self.__train_label = self.__train.iloc[:, 2:].copy(), self.__train.iloc[:, 1].copy()
        self.__test_feature, self.__test_id = self.__test.iloc[:, 1:].copy(), self.__test.iloc[:, [0]].copy()

        # 删除了所有以 ps_calc 开头的 feature
        self.__train_feature = self.__train_feature[[col for col in self.__train_feature.columns if not col.startswith("ps_calc_")]]
        self.__test_feature = self.__test_feature[self.__train_feature.columns]

        # train validation
        mask = int(len(self.__train_feature) * 0.8)
        self.__train_feature, self.__train_label, self.__validation_feature, self.__validation_label = (  # train_feature 名字用重了
            self.__train_feature[:mask].copy(),
            self.__train_label[:mask].copy(),
            self.__train_feature[mask:].copy(),
            self.__train_label[mask:].copy()
        )

        self.__numeric_columns = [col for col in self.__train_feature.columns if not col.endswith("_cat")]
        self.__categorical_columns = [col for col in self.__train_feature.columns if col.endswith("_cat")]

        # numeric
        for col in tqdm(self.__numeric_columns):
            self.__train_feature[col] = self.__train_feature[col].fillna(0)  # FIX BUG np.log1p 报错
            self.__validation_feature[col] = self.__validation_feature[col].fillna(0)
            self.__test_feature[col] = self.__test_feature[col].fillna(0)

        scaler = StandardScaler()
        scaler.fit(self.__train_feature[self.__numeric_columns])
        self.__train_feature[self.__numeric_columns] = scaler.transform(self.__train_feature[self.__numeric_columns])
        self.__validation_feature[self.__numeric_columns] = scaler.transform(self.__validation_feature[self.__numeric_columns])
        self.__test_feature[self.__numeric_columns] = scaler.transform(self.__test_feature[self.__numeric_columns])

        # categorical
        for col in tqdm(self.__categorical_columns):
            num_unique = self.__train_feature[col].nunique()

            if num_unique == 1:
                self.__train_feature = self.__train_feature.drop([col], axis=1)
                self.__validation_feature = self.__validation_feature.drop([col], axis=1)
                self.__test_feature = self.__test_feature.drop([col], axis=1)
                self.__categorical_columns.remove(col)
            else:

                if self.__train_feature[col].isna().sum():  # train 存在缺失
                    self.__train_feature[col] = self.__train_feature[col].fillna("missing")
                    mode = self.__train_feature[col].value_counts().index[0]
                    categories = self.__train_feature[col].unique()

                    self.__validation_feature[col] = self.__validation_feature[col].fiillna(mode)
                    self.__validation_feature[col] = [mode if i not in categories else i for i in self.__validation_feature[col]]
                    self.__test_feature[col] = self.__test_feature[col].fiillna(mode)
                    self.__test_feature[col] = [mode if i not in categories else i for i in self.__test_feature[col]]

                else:  # train 不存在缺失
                    mode = self.__train_feature[col].value_counts().index[0]
                    categories = self.__train_feature[col].unique()

                    self.__validation_feature[col] = self.__validation_feature[col].fillna(mode)
                    self.__validation_feature[col] = [mode if i not in categories else i for i in self.__validation_feature[col]]
                    self.__test_feature[col] = self.__test_feature[col].fillna(mode)
                    self.__test_feature[col] = [mode if i not in categories else i for i in self.__test_feature[col]]

                self.__train_feature[col] = self.__train_feature[col].astype(str)
                self.__validation_feature[col] = self.__validation_feature[col].astype(str)
                self.__test_feature[col] = self.__test_feature[col].astype(str)

                encoder = LabelEncoder()
                encoder.fit(self.__train_feature[col])
                self.__train_feature[col] = encoder.transform(self.__train_feature[col])
                self.__validation_feature[col] = encoder.transform(self.__validation_feature[col])
                self.__test_feature[col] = encoder.transform(self.__test_feature[col])
                self.__categorical_columns_num_unique[col] = len(encoder.classes_)

                self.__train_feature_for_model.append(self.__train_feature[col].values)
                self.__validation_feature_for_model.append(self.__validation_feature[col].values)
                self.__test_feature_for_model.append(self.__test_feature[col].values)

        self.__train_feature_for_model.append(self.__train_feature[self.__numeric_columns].values)
        self.__validation_feature_for_model.append(self.__validation_feature[self.__numeric_columns].values)
        self.__test_feature_for_model.append(self.__test_feature[self.__numeric_columns].values)

        self.__train_label = self.__train_label.values
        self.__validation_label = self.__validation_label.values

    def data_write(self):
        pickle.dump(self.__numeric_columns, file=open(os.path.join(self.__output_path, "numeric_columns.pkl"), mode="xb"))
        pickle.dump(self.__categorical_columns, file=open(os.path.join(self.__output_path, "categorical_columns.pkl"), mode="xb"))
        pickle.dump(self.__categorical_columns_num_unique,  file=open(os.path.join(self.__output_path, "categorical_columns_num_unique.pkl"), mode="xb"))

        pickle.dump(self.__train_feature_for_model, file=open(os.path.join(self.__output_path, "train_feature_for_model.pkl"), mode="xb"))
        pickle.dump(self.__validation_feature_for_model, file=open(os.path.join(self.__output_path, "validation_feature_for_model.pkl"), mode="xb"))
        pickle.dump(self.__test_feature_for_model, file=open(os.path.join(self.__output_path, "test_feature_for_model.pkl"), mode="xb"))

        pickle.dump(self.__train_label, file=open(os.path.join(self.__output_path, "train_label.pkl"), mode="xb"))
        pickle.dump(self.__validation_label, file=open(os.path.join(self.__output_path, "validation_label.pkl"), mode="xb"))


if __name__ == "__main__":
    eenn = EntityEmbeddingNeuralNetHyperas(
        input_path="E:\\Kaggle\\Porto Seguro Safe Driver Prediction",
        output_path="E:\\Kaggle\\Porto Seguro Safe Driver Prediction"
    )
    eenn.data_read()
    eenn.data_prepare()
    eenn.data_write()


