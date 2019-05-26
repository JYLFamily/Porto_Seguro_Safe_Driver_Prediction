# coding:utf-8

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from bayes_opt import BayesianOptimization


def gbm_cv(min_samples_leaf, subsample, max_features, learning_rate, n_estimators, feature, target):
    gbm = GradientBoostingClassifier(
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=7
    )
    val = cross_val_score(gbm, feature, target, scoring="roc_auc", cv=3)

    return val.mean()


def optimize_gbm(feature, target):
    def gbm_crossval(min_samples_leaf, subsample, max_features, learning_rate, n_estimators):
        return gbm_cv(
            min_samples_leaf=max(min(min_samples_leaf, 1.0), 0),
            subsample=max(min(subsample, 1.0), 0),
            max_features=max(min(max_features, 1.0), 0),
            learning_rate=max(min(learning_rate, 1.0), 0),
            n_estimators=max(int(round(n_estimators)), 1),
            feature=feature,
            target=target
        )

    optimizer = BayesianOptimization(
        f=gbm_crossval,
        pbounds={
            "min_samples_leaf": (0.05, 0.15),
            "subsample": (0.5, 1),
            "max_features": (0.5, 1),
            "learning_rate": (0.01, 0.05),
            "n_estimators": (100, 250),
        },
        random_state=7,
        verbose=2
    )

    optimizer.maximize(init_points=1, n_iter=1, alpha=1e-4)

    return optimizer.max["target"], optimizer.max["params"]