"""Classical baseline model builders for load forecasting."""

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def build_ridge_baseline(alpha=1.0, random_state=0):
    return Ridge(alpha=alpha, random_state=random_state)


def build_xgb_baseline(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=0,
):
    xgb_base = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective=objective,
        n_jobs=n_jobs,
        random_state=random_state
    )
    return MultiOutputRegressor(xgb_base)
