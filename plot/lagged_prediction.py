import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from tsl.utils import numpy_metrics

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

fig = plt.figure(figsize=[6.4, 3.2])
ax = plt.subplot(121)
observe_sensor = 0
forecasting_step = 0
# gat_tcn_hat = np.load('../data/lagged_gat_tcn_hat.npy')
tcn_hat = np.load('../data/tcn_hat.npy')
grin_y_true = np.load('../data/grin_y_true.npy')
###################预测自相关性###########################
tcn_rmse = numpy_metrics.rmse(tcn_hat[:, forecasting_step, :, :], grin_y_true[:, forecasting_step, :, :])
tcn_mae = numpy_metrics.mae(tcn_hat[:, forecasting_step, :, :], grin_y_true[:, forecasting_step, :, :])
tcn_mre = numpy_metrics.masked_mre(tcn_hat[:, forecasting_step, :, :], grin_y_true[:, forecasting_step, :, :])

t = np.arange(0, len(grin_y_true))
ax.plot(t, tcn_hat[:, 0, observe_sensor, :], color=color[0], label='TCN预测值', alpha=1)
ax.set_xlim(0, len(grin_y_true))
ax.plot(np.arange(0, len(grin_y_true)), grin_y_true[:, 0, observe_sensor, :], color=color[1], label='实际值')

ax.annotate('RMSE:%.2f' % tcn_rmse, xy=(0.65, 0.4), xycoords='axes fraction', font=font)
ax.annotate(' MRE:%.4f' % tcn_mre, xy=(0.65, 0.3), xycoords='axes fraction', font=font)
ax.annotate(' MAE:%.2f' % tcn_mae, xy=(0.65, 0.2), xycoords='axes fraction', font=font)

ax.legend(loc='center', bbox_to_anchor=(-0.0, 0.3, 1, 1), ncol=1, bbox_transform=ax.transAxes, prop=font1)
ax.grid(True, alpha=0.5)
ax.set_ylabel('渗压水位/m', font=font1)
ax.set_xlabel('历时/d', font=font1)
ax.set_title('(a)', y=-0.17, transform=ax.transAxes, font=font1)

###################预测自相关系数###########################
ax2 = inset_axes(ax, width="100%", height="100%", loc='lower left',
                 bbox_to_anchor=(1.1, -0.025, 1, 1), bbox_transform=ax.transAxes)
ax2.plot(t, tcn_hat[:, 0, observe_sensor, :], color=color[0])
ax2.plot(t, grin_y_true[:, 0, observe_sensor, :], color=color[1])
ax2.set_xlim(70, 100)
ax2.set_ylim(33, 39)
ax2.grid(True, alpha=0.5)
ax2.set_ylabel('渗压水位/m', font=font1)
ax2.set_xlabel('历时/d', font=font1)
mark_inset(ax, ax2, loc1=2, loc2=3, linestyle='--', fc="none", ec='r', lw=0.8)
ax2.set_title('(b)', y=-0.17, transform=ax2.transAxes, font=font1)

plt.subplots_adjust(wspace=0.2)
plt.savefig('./lagged_prediction.tiff', dpi=300, bbox_inches='tight')
plt.show()
pass