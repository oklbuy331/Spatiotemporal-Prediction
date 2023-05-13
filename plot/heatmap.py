import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd
import os

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
# df['P10'].plot(color='coral')

def add_right_cax(ax, pad, width):
    """
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    """
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)
    return cax


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# plot Pearson correlation imshow
im = axes[0].imshow(df.corr(), vmin=0, vmax=1, cmap='YlOrRd', aspect=1)
axes[0].set_xticks(np.arange(len(df.columns)))
axes[0].set_xticklabels(df.columns, rotation=90, fontdict=font)
axes[0].set_yticks(np.arange(len(df.columns)))
axes[0].set_yticklabels(df.columns, rotation=0, fontdict=font)
axes[0].tick_params('y', which='major', length=0)
axes[0].tick_params('x', which='major', length=0)
axes[0].set_title("(1) Pearson相关系数", x=0.5, y=1.02, ha='center', va='bottom', transform=axes[0].transAxes, font=font1)
# axes[0].get_xaxis().tick_top()

# plot mutual information imshow
mi = []
for i, sensor in enumerate(grin_hat):
    mi.append(mutual_info_regression(grin_hat, grin_hat[sensor]))
mi = np.array(mi)
mi = (mi - mi.min())/(mi.max() - mi.min())
im = axes[1].imshow(mi, vmin=0, vmax=1, cmap='YlOrRd', aspect=1)
axes[1].set_xticks(np.arange(len(df.columns)))
axes[1].set_xticklabels(df.columns, rotation=90, fontdict=font)
axes[1].set_yticks(np.arange(len(df.columns)))
axes[1].set_yticklabels(df.columns, rotation=0, fontdict=font)
axes[1].tick_params('y', which='major', length=0)
axes[1].tick_params('x', which='major', length=0)
axes[1].set_title("(2) 互信息", x=0.5, y=1.02, ha='center', va='bottom', transform=axes[1].transAxes, font=font1)
plt.subplots_adjust(wspace=0.0)
cax = add_right_cax(axes[1], pad=0.01, width=0.02)
cbar = fig.colorbar(im, cax=cax, pad=0.02)
cbar.ax.tick_params(which='major', length=0)
cbar.set_ticks(np.linspace(0, 1, 5),), cbar.set_label('相关程度', font=font1)
plt.savefig('./heatmap.tiff', dpi=300, bbox_inches='tight')
# plt.show()
pass
