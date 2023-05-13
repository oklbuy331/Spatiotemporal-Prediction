import pickle

import numpy as np
import pandas as pd
import os

import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tsl.utils import numpy_metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBRegressor
import scipy.stats as sp_stats

config = {
    'figure.figsize': [3.2, 3.2],
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.style': 'normal',
    'font.size': 9,
    'mathtext.fontset': 'stix',
    'font.weight': 'regular',
    'axes.labelpad': 1,
    'xaxis.labellocation': 'center',
    'yaxis.labellocation': 'center',
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'legend.frameon': False,
    'legend.labelspacing': 0.1,
    'legend.handletextpad': 0.4,
    'legend.loc': 'upper right',
    'legend.edgecolor': '1',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.unicode_minus': False,
}
plt.rcParams.update(config)

font = {'family': 'Times New Roman',
        'weight': 'regular',
        'style': 'normal',
        'size': 9,
        }

font1 = {'family': 'SimSun',
         'weight': 'regular',
         'style': 'normal',
         'size': 9}

color = ['coral', 'darkgreen', 'teal', 'darkblue', 'purple', 'crimson']

# r = sp_stats.expon(loc=1, scale=3).rvs(size=200)
# plt.hist(r, density=True, bins=15, histtype='stepfilled', alpha=0.2)
# plt.gca().set_xlim(r.min(), r.max())
# plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=15))
# plt.show()

data = pd.read_csv('../data/grin_hat_H.csv', index_col=0)
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
pwp = data.drop(['H'], axis=1)
level = data['H']
sensor = ['P01']
############################ standardize data ################################
scalers = {'pwp': StandardScaler(),
           'level': StandardScaler(),
           'sensor': StandardScaler()}
scalers['sensor'].fit(pwp[sensor].values.reshape(-1, 1))
pwp = pd.DataFrame(scalers['pwp'].fit_transform(pwp), index=data.index, columns=pwp.columns)
level = pd.DataFrame(scalers['level'].fit_transform(level.values.reshape(-1, 1)),
                     index=data.index, columns=['H'])


def feature_selection(df, level, use_pwp_avg=False):
    moving_avg = dict()
    windows = [3, 7, 14, 30, 45]
    if use_pwp_avg:
        for i, sensor in enumerate(df):
            serie = df[sensor].shift(1, fill_value=df[sensor][0])
            moving_sum_3 = serie.rolling(window=windows[0], axis=0).sum()
            moving_sum_7 = serie.rolling(window=windows[1], axis=0).sum()
            moving_sum_14 = serie.rolling(window=windows[2], axis=0).sum()
            moving_sum_30 = serie.rolling(window=windows[3], axis=0).sum()
            moving_sum_45 = serie.rolling(window=windows[4], axis=0).sum()
            moving_avg_1_3 = moving_sum_3 / 3.
            moving_avg_4_7 = (moving_sum_7 - moving_sum_3) / 4.
            moving_avg_7_14 = (moving_sum_14 - moving_sum_7) / 7.
            moving_avg_14_30 = (moving_sum_30 - moving_sum_14) / 16.
            moving_avg_30_45 = (moving_sum_45 - moving_sum_30) / 15.
            moving_avg_3 = serie.rolling(window=windows[0], axis=0).mean()
            moving_avg_7 = serie.rolling(window=windows[1], axis=0).mean()
            moving_avg_14 = serie.rolling(window=windows[2], axis=0).mean()
            moving_avg_30 = serie.rolling(window=windows[3], axis=0).mean()
            moving_avg_45 = serie.rolling(window=windows[4], axis=0).mean()
            moving_avg_H = pd.concat([serie, moving_avg_3, moving_avg_7, moving_avg_14, moving_avg_30,
                                      moving_avg_45], axis=1)
            moving_avg_H.columns = [sensor, sensor+'_3', sensor+'_7', sensor+'_14', sensor+'_30', sensor+'_45']
            moving_avg[sensor] = moving_avg_H

    # moving average reservoir level
    moving_sum_3 = level.rolling(window=windows[0], axis=0).sum()
    moving_sum_7 = level.rolling(window=windows[1], axis=0).sum()
    moving_sum_14 = level.rolling(window=windows[2], axis=0).sum()
    moving_sum_30 = level.rolling(window=windows[3], axis=0).sum()
    moving_sum_45 = level.rolling(window=windows[4], axis=0).sum()
    moving_avg_1_3 = moving_sum_3 / 3.
    moving_avg_4_7 = (moving_sum_7 - moving_sum_3) / 4.
    moving_avg_7_14 = (moving_sum_14 - moving_sum_7) / 7.
    moving_avg_14_30 = (moving_sum_30 - moving_sum_14) / 16.
    moving_avg_30_45 = (moving_sum_45 - moving_sum_30) / 15.
    moving_avg_3 = level.rolling(window=windows[0], axis=0).mean()
    moving_avg_7 = level.rolling(window=windows[1], axis=0).mean()
    moving_avg_14 = level.rolling(window=windows[2], axis=0).mean()
    moving_avg_30 = level.rolling(window=windows[3], axis=0).mean()
    moving_avg_45 = level.rolling(window=windows[4], axis=0).mean()
    moving_avg_level = pd.concat([level, moving_avg_3, moving_avg_7, moving_avg_14, moving_avg_30, moving_avg_45],
                                 axis=1)
    moving_avg_level.columns = ['level', 'level'+'_3', 'level'+'_7', 'level'+'_14', 'level'+'_30', 'level'+'_45']
    # time effect variables
    theta = pd.DataFrame(np.arange(len(level)) / 100, index=level.index, columns=['theta'])
    ln_theta = pd.DataFrame(np.log(np.arange(len(level)) + 1e-5 / 100), index=level.index, columns=['ln_theta'])
    predictors = pd.concat([moving_avg_level, theta, ln_theta], axis=1)
    if use_pwp_avg:
        for _, name in enumerate(moving_avg):
            predictors = pd.concat([predictors, moving_avg[name]], axis=1)
    return predictors


predictors = feature_selection(pwp, level, use_pwp_avg=False)
excluded = []
predictors.drop(excluded, axis=1, inplace=True)
predictors.drop(predictors.index[:44], inplace=True)
level.drop(predictors.index[:44], inplace=True)
pwp.drop(predictors.index[:44], inplace=True)


def model_fit_and_predict(model, inputs, targets, params_dist, n_iter=100):
    regr = model()
    split_idx = int(len(inputs) * 0.8)
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = RandomizedSearchCV(regr, params_dist, n_jobs=-1, n_iter=n_iter, scoring='neg_mean_absolute_error', )  # cv=kflod
    grid_result = grid_search.fit(inputs.iloc[:split_idx, :], targets[sensor][:split_idx].squeeze())
    print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))
    prediction = model(**grid_search.best_params_).fit(inputs.iloc[:split_idx, :], targets[sensor][:split_idx].squeeze()).predict(inputs.iloc[split_idx:, :])
    prediction = scalers['sensor'].inverse_transform(prediction.reshape(-1, 1))
    return grid_search.best_params_, prediction

#################################### model selection ##################################################

n_iter_search = 200
split_idx = int(len(predictors) * 0.8)

# XGBoost
learning_rate = [0.05]
max_depth = sp_stats.randint(3, 11)
min_child_weight = sp_stats.randint(1, 6)
gamma = np.arange(0.1, 0.2, 20)
subsample = sp_stats.uniform(0.5, 0.5)
colsample_bytree = sp_stats.uniform(0.5, 0.5)
n_estimators = sp_stats.randint(500, 1000)
reg_alpha = sp_stats.expon(loc=0, scale=5)
reg_lambda = sp_stats.expon(loc=0, scale=5)
param_dist_xgbr = dict(learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,
                        gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree, n_estimators=n_estimators,
                        reg_alpha=reg_alpha, reg_lambda=reg_lambda)
parameters_dict_xgbr, prediction_xgbr = model_fit_and_predict(XGBRegressor, predictors, pwp, param_dist_xgbr, n_iter=1000)
# 1. {'colsample_bytree': 0.8830720984541265, 'gamma': 0.1, 'learning_rate': 0.001, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 999, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.9444860993096162}
# 2.
xgbr = XGBRegressor(**parameters_dict_xgbr)

# MLP
hidden_layer_sizes = [[32, 64, 128],
                      [256],
                      [32, 64],
                      [32, 64, 128, 256],
                      [64, 128],
                      [32, 64, 32]]
learning_rate_init = [0.001, 0.01, 0.1]
learning_rate = ['constant', 'adaptive']
alpha = [0.0001, 0.0005, 0.001, 0.002]
early_stopping = [True]
param_dist_mlp = dict(hidden_layer_sizes=hidden_layer_sizes, learning_rate=learning_rate,
                      learning_rate_init=learning_rate_init, alpha=alpha, early_stopping=early_stopping)
parameters_dict_mlp, prediction_mlp = model_fit_and_predict(MLPRegressor, predictors, pwp, param_dist_mlp)
# 'learning_rate_init': 0.01, 'learning_rate': 'adaptive', 'hidden_layer_sizes': [32, 64, 128], 'early_stopping': True, 'alpha': 0.002
mlp = MLPRegressor(**parameters_dict_mlp)

# SVR
kernel = ['linear', 'rbf']
C = sp_stats.expon(loc=1, scale=5)
epsilon = sp_stats.gamma(a=2., scale=0.2)
param_dist_svr = dict(kernel=kernel, C=C, epsilon=epsilon)
parameters_dict_svr, prediction_svr = model_fit_and_predict(SVR, predictors, pwp, param_dist_svr, n_iter=1000)
# {'C': 6.135814029534986, 'epsilon': 0.01788052164404086, 'kernel': 'linear'}
svr = SVR(**parameters_dict_svr)

# lasso
alpha = sp_stats.expon(loc=0, scale=5)
fit_intercept = [True]
param_dist_lasso = dict(alpha=alpha, fit_intercept=fit_intercept)
parameters_dict_lasso, prediction_lasso = model_fit_and_predict(Lasso, predictors, pwp, param_dist_lasso, n_iter=1000)
# {'alpha': 0.49300983725652703, 'fit_intercept': True}
lasso = Lasso(**parameters_dict_lasso)

# random forest
max_depth = sp_stats.randint(1, 11)
n_estimators = sp_stats.randint(500, 1000)
min_samples_leaf = sp_stats.randint(1, 5)
min_samples_split = np.arange(2, 8, step=2)
param_dist_rf = dict(max_depth=max_depth, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
parameters_dict_rf, prediction_rf = model_fit_and_predict(RandomForestRegressor, predictors, pwp, param_dist_rf)
# {'max_depth': 9, 'min_samples_leaf': 4, 'min_samples_split': 6, 'n_estimators': 728}
rf = RandomForestRegressor(**parameters_dict_rf)


pwp = pd.DataFrame(scalers['pwp'].inverse_transform(pwp), index=predictors.index, columns=pwp.columns)
rmse_lasso = numpy_metrics.rmse(pwp[sensor][split_idx:].values, prediction_lasso)
mae_lasso = numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_lasso)
mre_lasso = numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_lasso)

############################ plot prediction #####################################
mae_lasso = numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_lasso)
mre_lasso = numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_lasso)
pass
