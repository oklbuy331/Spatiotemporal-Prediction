from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.fftpack import fft, ifft
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR, NuSVR

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

color = ['coral', 'darkgreen', 'teal', 'darkblue', 'purple']

font = {'family': 'Times New Roman',
        'weight': 'regular',
        'style': 'normal',
        'size': 9,
        }
font1 = {'family': 'SimSun',
         'weight': 'regular',
         'style': 'normal',
         'size': 9}

# Load model prediction
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

############################ ICA decomposition ################################
# ica = FastICA(n_components=pwp.shape[1], whiten="arbitrary-variance")
# S = pd.DataFrame(ica.fit_transform(pwp), index=data.index)
# split_idx = int(len(S) * 0.8)
# model = LinearRegression()
# model.fit(S.iloc[:split_idx, 0:-1], S.iloc[:split_idx, -1])
# prediction = model.predict(S.iloc[split_idx:, 0:-1])
# fig, ax = plt.subplots()
# ax.plot(prediction, c='k')
# ax.plot(S.iloc[split_idx:, -1].values, c='r')
# plt.show()
# prediction = ica.inverse_transform(prediction)
# pca = PCA(n_components=pwp.shape[1])
# H = pd.DataFrame(pca.fit_transform(pwp), index=data.index)
###########################################################
def feature_selection(level):
    windows = [3, 7, 14, 30, 45]
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
    # moving_avg_level = pd.concat(
    #     [level, moving_avg_1_3, moving_avg_4_7, moving_avg_7_14, moving_avg_14_30, moving_avg_30_45],
    #     axis=1)
    moving_avg_level = pd.concat([level, moving_sum_3, moving_sum_7, moving_sum_14, moving_sum_30, moving_sum_45],
                                 axis=1)
    moving_avg_level.columns = ['H', 'H_1_3', 'H_4_7', 'H_7_14', 'H_14_30', 'H_30_45']
    # time effect variables
    theta = pd.DataFrame(np.arange(len(level)) / 100, index=level.index, columns=['theta'])
    ln_theta = pd.DataFrame(np.log(np.arange(len(level)) / 100), index=level.index, columns=['ln_theta'])
    level_velocity = pd.DataFrame(level.diff().fillna(method='bfill').values, index=level.index, columns=['H_vel'])
    predictors = pd.concat([moving_avg_level, level_velocity, theta, ln_theta], axis=1)
    return predictors


predictors = feature_selection(level)
excluded = ['H_7_14', 'H_14_30', 'H_30_45', 'H_vel', 'theta']
predictors.drop(excluded, axis=1, inplace=True)
predictors.drop(predictors.index[:44], inplace=True)
level.drop(predictors.index[:44], inplace=True)
pwp.drop(predictors.index[:44], inplace=True)
svr = SVR(kernel='linear', C=0.4, epsilon=0.1)
split_idx = int(len(predictors) * 0.8)
svr.fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx])
prediction = svr.predict(predictors.iloc[split_idx:, :])
prediction_svr = scalers['sensor'].inverse_transform(prediction.reshape(-1, 1))

model = LinearRegression()
model.fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx])
prediction = model.predict(predictors.iloc[split_idx:, :])
prediction_lr = scalers['sensor'].inverse_transform(prediction.reshape(-1, 1))

gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                                 loss='squared_error').fit(predictors.iloc[:split_idx, :], pwp[sensor][:split_idx])
prediction = gbdt.predict(predictors.iloc[split_idx:, :])
prediction_gbdt = scalers['sensor'].inverse_transform(prediction.reshape(-1, 1))

pwp = pd.DataFrame(scalers['pwp'].inverse_transform(pwp), index=predictors.index, columns=pwp.columns)

rmse_svr = (mean_squared_error(pwp[sensor][split_idx:].values, prediction_svr))**0.5
mae_svr = mean_absolute_error(pwp[sensor][split_idx:].values, prediction_svr)

rmse_lr = (mean_squared_error(pwp[sensor][split_idx:].values, prediction_lr))**0.5
mae_lr = mean_absolute_error(pwp[sensor][split_idx:].values, prediction_lr)

rmse_gbdt = (mean_squared_error(pwp[sensor][split_idx:].values, prediction_gbdt))**0.5
mae_gbdt = mean_absolute_error(pwp[sensor][split_idx:].values, prediction_gbdt)

fig = plt.figure()
ax = plt.subplot(311)
ax.plot(prediction_svr, c='k')
ax.plot(pwp[sensor][split_idx:].values, c='r')
ax = plt.subplot(312)
ax.plot(prediction_lr, c='orange')
ax.plot(pwp[sensor][split_idx:].values, c='r')
ax = plt.subplot(313)
ax.plot(prediction_gbdt, c='orange')
ax.plot(pwp[sensor][split_idx:].values, c='r')
plt.show()

############################################################
rows, cols = len(sensor), 3
fig, axs = plt.subplots(rows, cols, figsize=[6.4, 6.4/cols])
for i, name in enumerate(sensor):
    y = data[name]
    y_diff = y.diff().fillna(method='bfill')
    t = np.arange(0, len(y))

    # ADF test
    dftest = adfuller(y_diff, autolag='AIC')
    print(name, '')
    dfountput = pd.Series(dftest[0:4], index=['ADF Test Statistic', 'p-value', 'lags used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfountput['Critical Value(%s)' % key] = value
    print(dfountput)

    axs[0].plot(np.arange(0, len(t)), y_diff.values, color=color[0], lw=1)

    lag_acf = acf(y_diff, nlags=365)
    axs[1].plot(np.arange(0, len(lag_acf)), lag_acf, color=color[1], lw=1)
    axs[1].axhline(y=-1.96/np.sqrt(len(y)), linestyle='--', color=color[1], lw=0.8, alpha=0.6)
    axs[1].axhline(y=1.96/np.sqrt(len(y)), linestyle='--', color=color[1], lw=0.8, alpha=0.6)

    fft_y = fft(y_diff.values)
    half_t = t[range(int(len(t) / 2))]
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    normalization_y = abs_y / len(t)  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(len(t) / 2))]  # 因为对称性，只取一半区间（单边频谱）
    axs[2].plot(half_t, normalization_half_y, color=color[2], lw=1)

axs[0].set_title('(a)库水位过程线', loc="center", y=-0.3, font=font1)
axs[0].set_xlabel('历时/d', font=font1)
axs[1].set_title('(b)自相关系数', loc="center", y=-0.3, font=font1)
axs[1].set_xlabel('滞后阶数', font=font1)
axs[2].set_title('(c)谱密度', loc="center", y=-0.3, font=font1)
axs[2].set_xlabel('频率/d', font=font1)
# fig.legend(prop=font1)
plt.savefig('autocorrelation.tiff', dpi=300, bbox_inches='tight')
plt.show()
pass


def test_arima(history, test_data, order):
    # scaler = MinMaxScaler()
    # history = scaler.fit_transform(history)
    # test_data = scaler.transform(test_data)
    prediction = []
    original = []
    error_list = []
    for t in range(len(test_data)):
        model = ARIMA(history, order=order[i]).fit()
        output = model.forecast()
        pred_value = output[0]
        original_value = test_data[t]
        history.append(original_value)
        error = ((abs(pred_value - original_value)) / original_value) * 100
        error_list.append(error)
        # print('predicted = %.2f, expected = %.2f, error = %.2f' % (pred_value, original_value, error), '%')
        prediction.append(float(pred_value))
        original.append(float(original_value))
    print('\n Mean Error in Prediction: %.2f' % (sum(error_list) / float(len(error_list))), '%')
    # prediction = scaler.inverse_transform(prediction)
    # original = scaler.inverse_transform(test_data)
    return prediction, original, error_list


# fig = plt.figure(figsize=[7.2, 6.4])
# for i, name in enumerate(sensor):
#     y = data[name]
#
#     test_size = int(0.8 * len(y))
#     train_data, test_data = y[0:test_size], y[test_size:]
#     history = [x for x in train_data]
#     order = [(1, 1, 1), (1, 1, 1)]
#
#     prediction, original, error = test_arima(history, test_data, order)
#     rol_mean = pd.Series.rolling(test_data, window=7).mean()
#     rol_std = pd.Series.rolling(test_data, window=7).std()
#
#     ax1 = plt.subplot(2, 2, 2*i + 1)
#     ax1.plot(test_data.index, prediction, label='Prediction', color=color[0], lw=1)
#     ax1.plot(test_data.index, original, label='Original', color=color[1], lw=1)
#
#     ax2 = plt.subplot(2, 2, 2*i + 2)
#     ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax2.invert_yaxis()
#     ax2.plot(test_data.index, rol_std, label='Prediction', color=color[2], lw=1)
#     ax3 = ax2.twinx()
#     ax3.plot(test_data.index, error, label='Relative Error', color=color[3], lw=1)
# plt.savefig('arima_forecasting.tiff', dpi=300, bbox_inches='tight')

print('Done')


pass
