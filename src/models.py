from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor
import pandas as pd


def train_model(df_train, y_train, df_test, id_):
    x_train, x_test, Y_train, y_test = train_test_split(df_train, y_train, train_size= 0.99, random_state = 42)

    Linear_re = LinearRegression(
        fit_intercept=True,
        copy_X=True,
        #tol=1e-06,
        n_jobs=None,
        positive=False,
    )

    Random_re = RandomForestRegressor(
        n_estimators=200,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=200,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        #monotonic_cst=None,
    )

    XGB_re = XGBRegressor()

    cat_re = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        task_type='CPU',
        devices='0'
    )

    LGB_re = LGBMRegressor(
        boosting_type = 'gbdt',
        num_leaves = 31,
        max_depth = -1,
        learning_rate = 0.1,
        n_estimators = 400,
        subsample_for_bin = 200000,
        class_weight = 'balanced',
        min_split_gain = 0.0,
        min_child_weight = 0.001,
        min_child_samples = 20,
        subsample = 1.0,
        subsample_freq = 0,
        colsample_bytree = 1.0,
        reg_alpha = 0.0,
        reg_lambda = 0.0,
        random_state = 42,
        n_jobs = None,
        importance_type = 'split',
    )

    models = [Linear_re, Random_re,XGB_re,cat_re, LGB_re]
    best_model = []
    for model in models:
        model.fit(x_train, Y_train)
        print(f"{model}")
        train_predict = model.predict(x_train)
        score = mean_absolute_error(Y_train, train_predict)
        print(f"{model}", score)
        test_predict = model.predict(x_test)
        score = mean_absolute_error(y_test, test_predict)
        print(f"{model}", mean_absolute_error(y_test, test_predict),"\n")
        best_model.append(score)
    
    test_predict = models[best_model.index(best_model.min())].predict(df_test)

    pred_test = models[1].predict(df_test)

    submission = pd.DataFrame({
        'id': id_,
        'Tm': pred_test
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission saved!")