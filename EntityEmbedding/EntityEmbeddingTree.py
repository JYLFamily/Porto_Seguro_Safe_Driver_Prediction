# coding:utf-8

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from bayes_opt import BayesianOptimization
np.random.seed(7)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


class EntityEmbeddingTree(BaseEstimator, TransformerMixin):
    def __init__(self, *, numeric_columns, categorical_columns):
        self.__numeric_columns = numeric_columns
        self.__categorical_columns = categorical_columns
        self.__target_encoder, self.__one_hot_encoder = [None for _ in range(2)]
        self.__clf = None

    def fit(self, X, y):
        X = X.copy(deep=True)
        y = y.copy(deep=True)

        self.__target_encoder = TargetEncoder()
        X[self.__numeric_columns] = X[self.__numeric_columns].fillna(-9999.0)
        X[self.__categorical_columns] = X[self.__categorical_columns].fillna("missing").astype(str)
        X[self.__categorical_columns] = self.__target_encoder.fit_transform(X[self.__categorical_columns], y)

        def __cv(
                min_samples_leaf,
                subsample, max_features,
                learning_rate, n_estimators):
            val = cross_val_score(
                GradientBoostingClassifier(
                    min_samples_leaf=max(min(min_samples_leaf, 1.0), 0),
                    subsample=max(min(subsample, 1.0), 0),
                    max_features=max(min(max_features, 1.0), 0),
                    learning_rate=max(min(learning_rate, 1.0), 0),
                    n_estimators=max(int(round(n_estimators)), 1),
                    random_state=7
                ),
                X,
                y,
                scoring="roc_auc",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            ).mean()

            return val

        params = {
            # tree parameter
            "min_samples_leaf": (0.05, 0.5),
            # bagging parameter
            "subsample": (0.5, 1.0),
            "max_features": (0.5, 1.0),
            # gradient boosting parameter
            "learning_rate": (0.001, 0.05),
            "n_estimators": (250, 500)
        }

        clf_bo = BayesianOptimization(__cv, params)
        clf_bo.maximize(init_points=5, n_iter=25, alpha=1e-4)
        max_score, max_param = 0, None
        for elem in clf_bo.res:
            if elem["target"] > max_score:
                max_score, max_param = elem["target"], elem["params"]

        self.__clf = GradientBoostingClassifier(
            min_samples_leaf=max(min(max_param["min_samples_leaf"], 1.0), 0),
            subsample=max(min(max_param["subsample"], 1.0), 0),
            learning_rate=max(min(max_param["learning_rate"], 1.0), 0),
            n_estimators=max(int(round(max_param["n_estimators"])), 1)
        )
        self.__clf.fit(X, y)

        return self

    def transform(self, X):
        X = X.copy(deep=True)

        X[self.__numeric_columns] = X[self.__numeric_columns].fillna(-9999.0)
        X[self.__categorical_columns] = X[self.__categorical_columns].fillna("missing").astype(str)
        X[self.__categorical_columns] = self.__target_encoder.transform(X[self.__categorical_columns])

        return self.__clf.apply(X).reshape(-1, self.__clf.n_estimators_)

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

