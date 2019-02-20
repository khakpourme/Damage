import xgboost
from sklearn.model_selection import GridSearchCV


#parameters to tune
testing_parameters = {
   'colsample_bytree': [0.4, 0.6, 0.8],
   'gamma': [0, 0.03, 0.1, 0.3],
   'min_child_weight': [1.5, 6, 10],
   'learning_rate': [0.1, 0.07],
   'max_depth': [3, 5],
   'n_estimators': [1000],
   'reg_alpha': [1e-5, 1e-2,  0.75],
   'reg_lambda': [1e-5, 1e-2, 0.45],
   'subsample': [0.6, 0.95]
}

xgbmod = xgboost.XGBRegressor()
gsearch1 = GridSearchCV(estimator = xgbmod, param_grid = testing_parameters)