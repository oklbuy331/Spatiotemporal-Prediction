import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tsl.utils import numpy_metrics
from scipy.stats import expon, gamma

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


plt.rc('font', family='Times New Roman')
x_data = ['1', '2', '3']
ax_data = [16, 28, 33]
ax1_data = [7, 8, 9]

fig = plt.figure(dpi=600)
ax = fig.add_subplot(111)

ax.set_ylim([15, 35])
ax.set_yticks = np.arange(15, 35)
ax.set_yticklabels = np.arange(15, 35)

bar_width = 0.3
ax.set_ylabel('this is y1', fontsize=18, fontweight='bold')
lns1 = ax.bar(x=np.arange(len(x_data)), width=bar_width, height=ax_data, label='y1', fc='steelblue', alpha=0.8)

for a, b in enumerate(ax_data):
    plt.text(a, b + 0.0005, '%s' % b, ha='center')

ax1 = ax.twinx()  # this is the important function

ax1.set_ylim([6, 10])
ax1.set_yticks = np.arange(6, 10)
ax1.set_yticklabels = np.arange(6, 10)
ax1.set_ylabel('this is y2', fontsize=18, fontweight='bold')
lns2 = ax1.bar(x=np.arange(len(x_data)) + bar_width, width=bar_width, height=ax1_data, label='y2', fc='indianred',
               alpha=0.8)

for a, b in enumerate(ax1_data):
    plt.text(a + 0.3, b + 0.001, '%s' % b, ha='center')

plt.xticks(np.arange(len(x_data)) + bar_width / 2, x_data)

ax.set_xlabel('double Y', fontsize=20, fontweight='bold')

fig.legend(loc=1, bbox_to_anchor=(0.28, 1), bbox_transform=ax.transAxes)
plt
plt.show()
pass