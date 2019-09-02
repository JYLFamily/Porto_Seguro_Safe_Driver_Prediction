# coding:utf-8

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization


def rf_cv(min_samples_leaf, n_estimators, feature, target):
    rf = RandomForestClassifier(
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators,
        n_jobs=2,
        random_state=7
    )
    val = cross_val_score(rf, feature, target, scoring="roc_auc", cv=3)

    return val.mean()


def optimize_rf(feature, target):
    def rf_crossval(min_samples_leaf, n_estimators):
        return rf_cv(
            min_samples_leaf=max(min(min_samples_leaf, 1.0), 0),
            n_estimators=max(int(round(n_estimators)), 1),
            feature=feature,
            target=target
        )

    optimizer = BayesianOptimization(
        f=rf_crossval,
        pbounds={
            "min_samples_leaf": (0.005, 0.05),
            "n_estimators": (15, 25),
        },
        random_state=7,
        verbose=2
    )

    optimizer.maximize(init_points=2, n_iter=2, alpha=1e-4)

    return optimizer.max["target"], optimizer.max["params"]