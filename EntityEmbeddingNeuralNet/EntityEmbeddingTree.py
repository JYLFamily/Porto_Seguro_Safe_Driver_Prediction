# coding:utf-8

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingClassifier
np.random.seed(7)
pd.set_option("display.max_row", None)
pd.set_option("display.max_columns", None)


class EntityEmbeddingTree(BaseEstimator, TransformerMixin):
    def __init__(self, *, numeric_columns, categorical_columns):
        self.__numeric_columns = numeric_columns
        self.__categorical_columns = categorical_columns
        self.__imputer = None
        self.__target_encoder = None
        self.__one_hot_encoder = None
        self.__clf = None

    def fit(self, X, y):
        X = X.copy(deep=True)
        y = y.copy(deep=True)

        self.__imputer = SimpleImputer(strategy="constant", fill_value=-9999.0)
        self.__target_encoder = TargetEncoder()

        X[self.__numeric_columns] = self.__imputer.fit_transform(X[self.__numeric_columns])
        X[self.__categorical_columns] = self.__target_encoder.fit_transform(X[self.__categorical_columns], y)

        self.__clf = GradientBoostingClassifier()
        self.__clf.fit(X, y)
        self.__clf.apply(X)

    def transform(self, X):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        pass




