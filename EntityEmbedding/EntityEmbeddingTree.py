# coding:utf-8

import os
import gc
import yaml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingClassifier
from EntityEmbedding.Util import optimize_gbm
np.random.seed(7)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


class EntityEmbeddingTree(BaseEstimator, TransformerMixin):
    def __init__(self, *, numeric_columns, categorical_columns):
        self.__numeric_columns = numeric_columns
        self.__categorical_columns = categorical_columns
        self.__target_encoder, self.__one_hot_encoder = [None for _ in range(2)]
        self.__max_target, self.__max_param = [None for _ in range(2)]
        self.__clf = None

    def fit(self, X, y):
        X = X.copy(deep=True)
        y = y.copy(deep=True)

        self.__target_encoder = TargetEncoder()
        X[self.__numeric_columns] = X[self.__numeric_columns].fillna(-9999.0)
        X[self.__categorical_columns] = X[self.__categorical_columns].fillna("missing").astype(str)
        X[self.__categorical_columns] = self.__target_encoder.fit_transform(X[self.__categorical_columns], y)

        # self.__max_target, self.__max_param = optimize_gbm(X, y)
        # self.__clf = GradientBoostingClassifier(
        #     min_samples_leaf=max(min(self.__max_param["min_samples_leaf"], 1.0), 0),
        #     subsample=max(min(self.__max_param["subsample"], 1.0), 0),
        #     learning_rate=max(min(self.__max_param["learning_rate"], 1.0), 0),
        #     n_estimators=max(int(round(self.__max_param["n_estimators"])), 1)
        # )
        self.__clf = GradientBoostingClassifier(
            min_samples_leaf=max(min(0.05, 1.0), 0),
            subsample=max(min(1.0, 1.0), 0),
            max_features=max(min(1.0, 1.0), 0),
            learning_rate=max(min(0.05, 1.0), 0),
            n_estimators=max(int(round(490.6)), 1)
        )

        self.__clf.fit(X, y)
        gc.collect()

        return self

    def transform(self, X):
        X = X.copy(deep=True)

        X[self.__numeric_columns] = X[self.__numeric_columns].fillna(-9999.0)
        X[self.__categorical_columns] = X[self.__categorical_columns].fillna("missing").astype(str)
        X[self.__categorical_columns] = self.__target_encoder.transform(X[self.__categorical_columns])
        gc.collect()

        return self.__clf.apply(X).reshape(-1, self.__clf.n_estimators_)[:, :10]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X=X, y=y)

        return self.transform(X)


if __name__ == "__main__":
    with open("config.yaml", encoding="UTF-8") as config_file:
        config = yaml.load(config_file)

    train = pd.read_csv(os.path.join(config["input_path"], "train.csv"), na_values=-1, nrows=500)
    test = pd.read_csv(os.path.join(config["input_path"], "test.csv"), na_values=-1, nrows=500)

    train_feature, train_label = train.iloc[:, 2:].copy(), train.iloc[:, 1].copy()
    test_feature = test.iloc[:, 1:].copy()
    del train, test

    train_feature = train_feature[[col for col in train_feature.columns if not col.startswith("ps_calc_")]]
    test_feature = test_feature[train_feature.columns]

    ncs = [col for col in train_feature.columns if not col.endswith(("_bin", "_cat"))]
    ccs = [col for col in train_feature.columns if col.endswith(("_bin", "_cat"))]

    eet = EntityEmbeddingTree(numeric_columns=ncs, categorical_columns=ccs)
    eet.fit(X=train_feature, y=train_label)
    print(eet.transform(X=train_feature).shape)
    print(eet.transform(X=test_feature).shape)

