import copy
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tsl
import yaml
from matplotlib import pyplot as plt
from torch import nn
from tsl import config
from tsl.data import SpatioTemporalDataModule, ImputationDataset, SpatioTemporalDataset, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.imputers import Imputer
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers.graph_convs import DiffConv
from tsl.nn.models import RNNModel, TCNModel
from tsl.nn.models.imputation import GRINModel
from tsl.nn.models.stgn import STCNModel, DCRNNModel
from tsl.nn.utils import casting
from tsl.ops.connectivity import edge_index_to_adj
from tsl.ops.imputation import add_missing_values, sample_mask
from tsl.predictors import Predictor
from tsl.utils import ArgParser, parser_utils, numpy_metrics
from tsl.utils.python_utils import ensure_list
from einops.layers.torch import Rearrange

from data.porewaterpressure import PoreWaterPressure
from mtad_gat import MTAD_GAT


class TimeThenSpaceModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 n_nodes: int,
                 horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = StaticGraphEmbedding(n_nodes, hidden_size)
        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',)
        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb, return_last_state=True)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128, 256])
        parser.add_argument('--u-size', type=int, default=None)
        return parser


def get_model_classes(model_str):
    if model_str == 'timethenspace':
        model = TimeThenSpaceModel
    elif model_str == 'rnn':
        model = RNNModel
    elif model_str == 'gat_tcn':
        model = MTAD_GAT
    elif model_str == 'tcn':
        model = TCNModel
    elif model_str == 'stcn':
        model = STCNModel
    elif model_str == 'dcrnn':
        model = DCRNNModel
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument("--model-name", type=str, default='tcn')
    parser.add_argument("--dataset-name", type=str, default='grin')
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--config", type=str, default='inference.yaml')
    parser.add_argument("--root", type=str, default='log')

    # dataset setting
    parser.add_argument('--use-exogenous', type=bool, default=False)

    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--batches-epoch', type=int, default=300)
    parser.add_argument('--batch-inference', type=int, default=32)
    parser.add_argument('--split-batch-in', type=int, default=1)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)
    parser.add_argument('--monitor-metric', type=str, default='train_loss')

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.6)

    known_args, _ = parser.parse_known_args()
    model_cls = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def load_model(exp_dir, exp_config, dm):
    model_cls = get_model_classes(exp_config['model_name'])
    additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                    input_size=dm.n_channels,
                                    output_size=dm.n_channels,
                                    window_size=dm.window)

    model_kwargs = parser_utils.filter_args(
        args={**exp_config, **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # setup imputer

    # setup predictor model_cls
    predictor = Predictor(
        model_class=model_cls,              # our initialized model
        optim_class=torch.optim.Adam,  # specify optimizer to be used...
        optim_kwargs={'lr': 0.001},    # ...and parameters for its initialization
        model_kwargs=model_kwargs,
        loss_fn=None,               # which loss function to be used
    )

    model_path = None
    for file in os.listdir(exp_dir):
        if file.endswith(".ckpt"):
            model_path = os.path.join(exp_dir, file)
            break
    if model_path is None:
        raise ValueError(f"Model not found.")

    predictor.load_model(model_path)
    predictor.freeze()
    return predictor


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def run_experiment(args):
    # Set configuration
    args = copy.deepcopy(args)
    tsl.logger.disabled = True

    ########################################
    # load config                          #
    ########################################
    if args.root is None:
        root = tsl.config.log_dir
    else:
        root = os.path.join(tsl.config.curr_dir, args.root)
    exp_dir = os.path.join(root, args.dataset_name,
                           args.model_name, args.exp_name)

    with open(os.path.join(exp_dir, 'config.yaml'), 'r') as fp:
        exp_config = yaml.load(fp, Loader=yaml.FullLoader)

    ########################################
    # load dataset                         #
    ########################################

    dataset = PoreWaterPressure(root='./data', dataset_name=args.dataset_name)
    dataset.df.iloc[1957, 0] = dataset.df.iloc[1957, 0] - 2.
    # set exogenous map and batch map
    if args.use_exogenous:
        predictors = dataset.predictors
        level_vars = pd.concat([dataset.level, dataset.level_velocity], axis=1)
        exog_map = {'global_level_vars': level_vars}
        input_map = {
            'u': 'level_vars',
            'x': 'predictors'
        }
    else:
        level_vars = dataset.predictors
        exog_map = {'global_predictors': level_vars}
        input_map = {
            'x': 'data'
        }

    ########################################
    # load data module                     #
    ########################################
    connectivity = dataset.get_connectivity(threshold=args.adj_threshold,
                                            include_self=False,
                                            normalize_axis=1,
                                            layout="edge_index")
    if args.model_name == 'gat_tcn':
        connectivity = None

    torch_dataset = SpatioTemporalDataset(dataset.dataframe(),
                                          connectivity=connectivity,
                                          exogenous=exog_map,
                                          input_map=input_map,
                                          mask=dataset.mask,
                                          delay=0,
                                          horizon=exp_config['horizon'],
                                          window=exp_config['window'],
                                          stride=exp_config['stride'])

    # Normalize data using mean and std computed over time and node dimensions
    if args.use_exogenous:
        scalers = {'data': StandardScaler(axis=0),
                   'predictors': StandardScaler(axis=0)
                   }
    else:
        scalers = {'data': StandardScaler(axis=0),
                   'predictors': StandardScaler(axis=0)}

    # Split data sequentially:
    #   |------------ dataset -----------|
    #   |--- train ---|- val -|-- test --|
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=args.batch_inference,
    )
    dm.setup()

    ########################################
    # load model                           #
    ########################################

    predictor = load_model(exp_dir, exp_config, dm)

    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()))

    ########################################
    # inference                            #
    ########################################

    seeds = ensure_list(args.test_mask_seed)
    mae = []
    mse = []
    mre = []

    train_output = trainer.predict(predictor, dataloaders=dm.train_dataloader(shuffle=False))
    val_output = trainer.predict(predictor, dataloaders=dm.val_dataloader(shuffle=False))
    test_output = trainer.predict(predictor, dataloaders=dm.test_dataloader(shuffle=False))

    # y_hat = torch.cat([train_output['y_hat'], val_output['y_hat'], test_output['y_hat']], dim=0)[:, train_output['y_hat'].shape[1]-1, :].squeeze(-1)
    # y_hat = casting.numpy(y_hat)
    y_hat = casting.numpy(test_output['y_hat'])

    # y_true = torch.cat([train_output['y'], val_output['y'], test_output['y']], dim=0)[:, train_output['y'].shape[1]-1, :].squeeze(-1)
    # y_true = casting.numpy(y_true)
    y_true = casting.numpy(test_output['y'])
    # fig, ax = plt.subplots(1, 1, figsize=[7.5, 3.2])
    # forecasting_step = 0
    # observe_sensor = 0
    # t = np.arange(0, len(y_hat))
    # ax.plot(t, y_hat[:, 0, observe_sensor, :], color='orange', label='TimeThenSpace预测')
    # ax.plot(t, y_true[:, 0, observe_sensor, :], color='k', label='TimeThenSpace预测')
    # plt.show()
    # y_hat = [dataset.ica.inverse_transform(y_hat[:, i, :, 0]) for i in range(y_hat.shape[1])]
    # y_hat = np.array(y_hat).transpose([1, 0, 2])
    # y_hat = np.expand_dims(y_hat, axis=-1)
    # y_true = [dataset.ica.inverse_transform(y_true[:, i, :, 0]) for i in range(y_true.shape[1])]
    # y_true = np.array(y_true).transpose([1, 0, 2])
    # y_true = np.expand_dims(y_true, axis=-1)
    np.save('./data/%s_hat.npy' % args.model_name, y_hat)
    np.save('./data/%s_y_true.npy' % args.dataset_name, y_true)

    # mask = torch.cat([train_output['mask'], val_output['mask'], test_output['mask']], dim=0)[:, train_output['mask'].shape[1]-1, :].squeeze(-1)
    # mask = casting.numpy(mask)
    mask = casting.numpy(test_output['mask'])

    # evaluate model

    check_mae = numpy_metrics.masked_mae(y_hat[:, 0, :], y_true[:, 0, :], mask[:, 0, :])
    for i in range(y_true.shape[2]):
        check_mae = np.append(check_mae, numpy_metrics.masked_mae(y_hat[:, 0, i], y_true[:, 0, i], mask[:, 0, i]))
    mae.append(check_mae)

    check_mse = numpy_metrics.masked_mse(y_hat[:, 0, :], y_true[:, 0, :], mask[:, 0, :])
    for i in range(y_true.shape[2]):
        check_mse = np.append(check_mse, numpy_metrics.masked_mse(y_hat[:, 0, i], y_true[:, 0, i], mask[:, 0, i]))
    mse.append(check_mse)

    check_mre = numpy_metrics.masked_mre(y_hat[:, 0, :], y_true[:, 0, :], mask[:, 0, :])
    for i in range(y_true.shape[2]):
        check_mre = np.append(check_mre, numpy_metrics.masked_mre(y_hat[:, 0, i], y_true[:, 0, i], mask[:, 0, i]))
    mre.append(check_mre)
    print(count_param(predictor))
    print(f'MAE over {len(seeds)} runs: {np.mean(mae):.2f}±{np.std(mae):.2f}')


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
