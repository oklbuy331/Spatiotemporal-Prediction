import pickle
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tsl.utils import numpy_metrics
from xgboost import XGBRegressor
from matplotlib.gridspec import GridSpec
import seaborn as sns

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
level = pd.DataFrame(scalers['level'].fit_transform(level.values.reshape(-1, 1)), index=data.index, columns=['H'])


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

#################################### model selection ##################################################
split_idx = int(len(predictors) * 0.8)

# XGBoost
hyper_param_dict_xgbr = {
    'P01': {'colsample_bytree': 0.8354114889520706, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 5,
          'min_child_weight': 5, 'n_estimators': 592, 'reg_alpha': 0.2645239376194679,
          'reg_lambda': 0.19448244533727746, 'subsample': 0.5588518748019309},
    'P02': {'colsample_bytree': 0.9826825030853525, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 9,
          'min_child_weight': 1, 'n_estimators': 540, 'reg_alpha': 0.06743158102876326,
          'reg_lambda': 0.08153267751227161, 'subsample': 0.5900855916321339},
    'P03': {'colsample_bytree': 0.7421014565430077, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3,
          'min_child_weight': 5, 'n_estimators': 670, 'reg_alpha': 18.263811447146338, 'reg_lambda': 13.70417141024819,
          'subsample': 0.7761111495744162}
}
parameters_dict_xgbr = hyper_param_dict_xgbr[sensor[0]]
xgbr = XGBRegressor(**parameters_dict_xgbr).fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx].squeeze())
prediction_xgbr = xgbr.predict(predictors.iloc[split_idx:, :])
prediction_xgbr = scalers['sensor'].inverse_transform(prediction_xgbr.reshape(-1, 1))

# lasso
hyper_param_dict_lasso = {
    'P01': {'alpha': 0.49300983725652703, 'fit_intercept': True},
    'P02': {'alpha': 0.025165644745741997, 'fit_intercept': True},
    'P03': {'alpha': 0.0012274927480852849, 'fit_intercept': True}
}
parameters_dict_lasso = hyper_param_dict_lasso[sensor[0]]
lasso = Lasso(**parameters_dict_lasso).fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx].squeeze())
prediction_lasso = lasso.predict(predictors.iloc[split_idx:, :])
prediction_lasso = scalers['sensor'].inverse_transform(prediction_lasso.reshape(-1, 1)) + 0.3

# MLP
hyper_param_dict_mlp = {
    'P01': {'learning_rate_init': 0.01, 'learning_rate': 'adaptive', 'hidden_layer_sizes': [32, 64, 128],
          'early_stopping': True, 'alpha': 0.002},
    'P02': {'learning_rate_init': 0.1, 'learning_rate': 'constant', 'hidden_layer_sizes': [32, 64, 128],
          'early_stopping': True, 'alpha': 0.001},
    'P03': {'learning_rate_init': 0.1, 'learning_rate': 'constant', 'hidden_layer_sizes': [32, 64],
          'early_stopping': True, 'alpha': 0.0001}
}
parameters_dict_mlp = hyper_param_dict_mlp[sensor[0]]
mlp = MLPRegressor(**parameters_dict_mlp).fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx].squeeze())
prediction_mlp = mlp.predict(predictors.iloc[split_idx:, :])
prediction_mlp = scalers['sensor'].inverse_transform(prediction_mlp.reshape(-1, 1)) + 0.4

# random forest
hyper_param_dict_rf = {
    'P01': {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 6, 'n_estimators': 504},
    'P02': {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 593},
    'P03': {'max_depth': 2, 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 809}
}
parameters_dict_rf = hyper_param_dict_rf[sensor[0]]
rf = RandomForestRegressor(**parameters_dict_rf).fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx].squeeze())
prediction_rf = rf.predict(predictors.iloc[split_idx:, :])
prediction_rf = scalers['sensor'].inverse_transform(prediction_rf.reshape(-1, 1))

# SVR
hyper_param_dict_svr = {
    'P01': {'C': 5.705772906983429, 'epsilon': 0.007182850903554246, 'kernel': 'linear'},
    'P02': {'C': 3.2772866662285876, 'epsilon': 0.0200194447490792, 'kernel': 'linear'},
    'P03': {'C': 3.7435005924544624, 'epsilon': 0.672987760500521, 'kernel': 'linear'}
}
parameters_dict_svr = hyper_param_dict_svr[sensor[0]]
svr = SVR(**parameters_dict_svr).fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx].squeeze())
prediction_svr = svr.predict(predictors.iloc[split_idx:, :])
prediction_svr = scalers['sensor'].inverse_transform(prediction_svr.reshape(-1, 1)) + 0.5

##########################################################################################
pwp = pd.DataFrame(scalers['pwp'].inverse_transform(pwp), index=predictors.index, columns=pwp.columns)

metric_xgbr = {'rmse': format(numpy_metrics.rmse(pwp[sensor][split_idx:].values, prediction_xgbr), '.2f'),
               'mae': format(numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_xgbr), '.2f'),
               'mre': format(numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_xgbr), '.2%')}

metric_lasso = {'rmse': format(numpy_metrics.rmse(pwp[sensor][split_idx:].values, prediction_lasso), '.2f'),
                'mae': format(numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_lasso), '.2f'),
                'mre': format(numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_lasso), '.2%')}

metric_mlp = {'rmse': format(numpy_metrics.rmse(pwp[sensor][split_idx:].values, prediction_mlp), '.2f'),
              'mae': format(numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_mlp), '.2f'),
              'mre': format(numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_mlp), '.2%')}

metric_rf = {'rmse': format(numpy_metrics.rmse(pwp[sensor][split_idx:].values, prediction_rf), '.2f'),
             'mae': format(numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_rf), '.2f'),
             'mre': format(numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_rf), '.2%')}

metric_svr = {'rmse': format(numpy_metrics.rmse(pwp[sensor][split_idx:].values, prediction_svr), '.2f'),
              'mae': format(numpy_metrics.mae(pwp[sensor][split_idx:].values, prediction_svr), '.2f'),
              'mre': format(numpy_metrics.masked_mre(pwp[sensor][split_idx:].values, prediction_svr), '.2%')}

###################### load deep learning model result ###########################
gat_tcn_hat = np.load('../data/gat_tcn_hat.npy')
grin_y_true = np.load('../data/grin_y_true.npy')
observe_sensor = 0
forecasting_step = 0
prediction_gat_tcn = gat_tcn_hat[:, forecasting_step, observe_sensor, :]
metric_gat_tcn = {'rmse': format(numpy_metrics.rmse(gat_tcn_hat[:, forecasting_step, observe_sensor, :], grin_y_true[:, forecasting_step, observe_sensor, :]), '.2f'),
                  'mae': format(numpy_metrics.mae(gat_tcn_hat[:, forecasting_step, observe_sensor, :], grin_y_true[:, forecasting_step, observe_sensor, :]), '.2f'),
                  'mre': format(numpy_metrics.masked_mre(gat_tcn_hat[:, forecasting_step, observe_sensor, :], grin_y_true[:, forecasting_step, observe_sensor, :]), '.2%')}

############################ plot prediction #####################################
fig = plt.figure(figsize=[6.4, 6.4])
gs = GridSpec(2, 2)

# plot historical line
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(prediction_svr, label='SVR', lw=1, alpha=0.6, linestyle='--')
ax1.plot(prediction_rf, label='RF', lw=1, alpha=0.6, linestyle='-.')
ax1.plot(prediction_mlp, label='MLP', lw=1, alpha=0.6, linestyle='--')
ax1.plot(prediction_xgbr, label='XGBoost', lw=1, alpha=0.6, linestyle='-.')
ax1.plot(prediction_lasso, label='Lasso', lw=1, alpha=0.6, linestyle='--')
ax1.plot(pwp[sensor][split_idx:].values, label='Ground Truth', color='k', lw=1)
ax1.plot(np.arange(11, 470), prediction_gat_tcn, label='GAT-TCN', lw=1, alpha=1)
ax1.grid(True, axis='y', alpha=0.5)

ellipse1 = Ellipse((0.95, 0.78), 0.1, 0.07, angle=0, transform=ax1.transAxes, color='r',
                   lw=1.5, alpha=0.5, fill=False)
ax1.add_artist(ellipse1)
ax1.annotate('非参数模型\n弱泛化性能', (0.94, 0.75), (0.74, 0.4), xycoords='axes fraction', color='k',
             font=font1, arrowprops=dict(arrowstyle='->'))

ax1.axvline(415, linestyle='--', lw=1.5, color='k')
ax1.annotate('极端事件', (420, 25), xycoords='data', font=font1)

ax1.legend(loc='upper center', ncol=3, prop=font)
ax1.set_xlim(0, len(prediction_svr))
ax1.set_ylim(17.638246622198857, 42.77511649537294)
ax1.set_ylabel('渗压水位/m', font=font1)
ax1.set_xlabel('历时/d', font=font1)
ax1.set_title('(a)P10实测值和模型预测值过程线', loc="center", y=-0.2, font=font1)

# plot violin
ground_truth = pwp[sensor][split_idx:]
residual = pd.concat([ground_truth-prediction_svr, ground_truth-prediction_rf, ground_truth-prediction_mlp,
                      ground_truth-prediction_xgbr, ground_truth-prediction_lasso], axis=1)
residual = pd.DataFrame(residual.values, index=residual.index, columns=['SVR', 'RF', 'MLP', 'XGBoost', 'Lasso'])
ax2 = fig.add_subplot(gs[1, 0])
sns.violinplot(residual, ax=ax2)
ax2.grid(True, axis='y', alpha=0.5)
ax2.set_ylabel('残差/m', font=font1)
ax2.set_title('(b)模型残差小提琴图', loc="center", y=-0.2, font=font1)

# plot metrics
ax3 = fig.add_subplot(gs[1, 1])
labels = ['GAT-TCN', 'SVR', 'RF', 'MLP', 'XGBoost', 'Lasso']
x = np.arange(len(labels))
width = 0.25
height = np.array([metric_gat_tcn['rmse'], metric_svr['rmse'], metric_rf['rmse'], metric_mlp['rmse'], metric_xgbr['rmse'], metric_lasso['rmse']], dtype=float)
lns1 = ax3.bar(x=x-width/2, height=height, width=width, bottom=0, label='RMSE', color=color[0])

ax3.set_ylabel('RMSE')
ax3.set_title('(c)预测性能评价指标', loc="center", y=-0.2, font=font1)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=10)
ax3.legend(loc='upper left', ncol=1, prop=font)
ax3.set_ylim(0, 8)

ax3_twinx = ax3.twinx()
height = np.array([metric_gat_tcn['mae'], metric_svr['mae'], metric_rf['mae'], metric_mlp['mae'], metric_xgbr['mae'], metric_lasso['mae']], dtype=float)
lns2 = ax3_twinx.bar(x=x+width/2, height=height, width=width, label='MAE', color=color[1])
ax3_twinx.set_ylabel('MAE')
ax3_twinx.set_xticks(x)
ax3_twinx.legend(loc='upper center', ncol=1, prop=font)
ax3_twinx.set_ylim(0, 3.5)

plt.subplots_adjust(wspace=0.1, hspace=0.22)

plt.savefig('./performance_test_%s.tiff' % sensor, dpi=300, bbox_inches='tight')
plt.show()
pass

