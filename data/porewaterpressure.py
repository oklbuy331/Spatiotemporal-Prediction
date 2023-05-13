import os
from typing import Optional, Sequence, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from tsl import logger
from datetime import datetime

from tsl.data.datamodule.splitters import AtTimeStepSplitter
from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes import PandasDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.data.utils import HORIZON
from seaborn import heatmap


class PoreWaterPressure(PandasDataset, MissingValuesMixin):
    """

    """
    similarity_options = {'distance', 'Pearson', 'mutual_information', 'Granger'}
    temporal_aggregation_options = {'mean', 'nearest'}
    spatial_aggregation_options = {'mean'}

    def __init__(self, root: str = None,
                 similarity_score: str = 'Pearson',
                 dataset_name: str = 'grin',
                 impute_nans: bool = True,
                 freq: Optional[str] = None):
        self.root = root
        self.dataset_name = dataset_name
        df, self.predictors, mask, dist = self.load()
        # self.level_velocity = pd.DataFrame(self.level.diff().fillna(method='bfill').values, index=df.index, columns=['H_vel'])
        super().__init__(dataframe=df,
                         attributes=dict(dist=dist),
                         mask=mask,
                         freq=freq,
                         similarity_score=similarity_score,
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='pwp',
                         name='PWP')

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.csv']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['pwp_dist.npy']

    def load_raw(self):
        data = pd.read_csv(self.root + '/%s_hat_H.csv' % self.dataset_name, index_col=0)  # pore water pressure observations
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        dist = np.load(self.root + '/dist.npy')
        df = data.drop(['H'], axis=1)
        level = data['H']
        # rising = np.repeat(np.linspace(0, 4, 10).reshape(-1, 1), 17, axis=1)
        # df.iloc[-10:, :] = df.iloc[-10:, :] + rising
        return df, level, dist

    def feature_selection(self, df, level, use_pwp_avg=False):
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
                velocity = serie.diff().fillna(method='bfill')
                moving_avg_H = pd.concat([serie, moving_avg_3, moving_avg_7, moving_avg_14, moving_avg_30,
                                          moving_avg_45], axis=1) # velocity
                # moving_avg_level = pd.concat([level, moving_sum_3, moving_sum_7, moving_sum_14, moving_sum_30, moving_sum_45],
                #                              axis=1)
                moving_avg_H.columns = [sensor, sensor+'_3', sensor+'_7', sensor+'_14', sensor+'_30',
                                        sensor+'_45', ] # sensor+'_vel'
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
        # moving_avg_level = pd.concat([level, moving_sum_3, moving_sum_7, moving_sum_14, moving_sum_30, moving_sum_45],
        #                              axis=1)
        moving_avg_level.columns = ['level', 'level'+'_3', 'level'+'_7', 'level'+'_14', 'level'+'_30', 'level'+'_45']
        # time effect variables
        theta = pd.DataFrame(np.arange(len(level))/100, index=level.index, columns=['theta'])
        ln_theta = pd.DataFrame(np.log(np.arange(len(level))/100), index=level.index, columns=['ln_theta'])
        level_velocity = pd.DataFrame(level.diff().fillna(method='bfill').values, index=level.index, columns=['H_vel'])
        predictors = pd.concat([moving_avg_level, level_velocity, theta, ln_theta], axis=1)  #
        if use_pwp_avg:
            for _, name in enumerate(moving_avg):
                predictors = pd.concat([predictors, moving_avg[name]], axis=1)
        return predictors

    def load(self, use_pca=False, uniq_output=False):
        # load readings and stations metadata
        df, level, dist = self.load_raw()
        # feature selection
        predictors = self.feature_selection(df, level)
        # inject anomalies
        heatmap(predictors.corr(), cbar=True)
        # plt.show()
        # exclude specified variables from inputs
        excluded = ['H_vel', 'level'+'_3', 'level'+'_7', 'level'+'_14', 'level'+'_30', 'level'+'_45',]  #  'theta', 'ln_theta'
        predictors.drop(excluded, axis=1, inplace=True)
        # drop nan
        predictors.drop(df.index[:44], inplace=True)
        if use_pca:
            self.pca = PCA(n_components=14)
            predictors = pd.DataFrame(self.pca.fit_transform(predictors), index=predictors.index)
        df.drop(df.index[:44], inplace=True)
        if uniq_output:
            df = pd.DataFrame(df['P01'].values, index=predictors.index, columns=['P01'])
        # compute the masks:
        mask = (~pd.isna(df.values)).astype('uint8')  # 1 if value is valid
        return df, predictors, mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'pwp':
            first_val_ts = datetime.strptime("2019-10-10", "%Y-%m-%d")
            first_test_ts = datetime.strptime("2020-08-10", "%Y-%m-%d")
            return AtTimeStepSplitter(first_val_ts=first_val_ts,
                                      first_test_ts=first_test_ts)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            # calculate similarity in terms of spatial distance
            theta = np.std(self.dist)
            return gaussian_kernel(self.dist, theta=theta)

        elif method == "Pearson":
            # calculate similarity in terms of Pearson correlation
            return self.predictors.corr().values

        elif method == "mutual_information":
            # calculate similarity in terms of mutual information between sensor records
            mi = np.load(os.getcwd() + '\\spin\\datasets\\mi.npy')
            for i in range(len(mi)):
                min = mi[i].min(); max = mi[i].max()
                mi[i] = (mi[i] - min)/(max - min)
            return mi

        else:
            # calculate similarity in terms of Granger causalty
            theta = np.std(self.dist[:36, :36])
            return gaussian_kernel(self.dist, theta=theta)
