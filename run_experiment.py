# Install required packages.
import copy
import datetime
import os

import tsl
import torch
import numpy as np
import pandas as pd
import yaml
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.layers.graph_convs import DiffConv
from tsl.nn.layers.norm import Norm
from tsl.nn.metrics import MaskedMAE, MaskedMAPE, MaskedMSE
from tsl.nn.models import RNNModel
from tsl.nn.models.stgn import STCNModel, DCRNNModel
from tsl.nn.ops.ops import Lambda
from tsl.nn.utils import casting, get_layer_activation
from tsl.predictors import Predictor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tsl.utils import parser_utils, numpy_metrics
from tsl.utils.parser_utils import ArgParser, str_to_bool
from data.porewaterpressure import PoreWaterPressure
import torch.nn as nn
from tsl.nn.blocks.encoders import RNN, ConditionalBlock, TemporalConvNet
from einops.layers.torch import Rearrange
from mtad_gat import *
from tsl.nn.models import TCNModel


class MyTCN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 output_size,
                 horizon,
                 kernel_size,
                 n_layers,
                 exog_size,
                 readout_kernel_size=1,
                 resnet=True,
                 dilation=1,
                 activation='relu',
                 n_convs_layer=2,
                 dropout=0.,
                 norm="none",
                 gated=False):
        super(MyTCN, self).__init__()

        if exog_size > 0:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  dropout=dropout,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        layers = []
        self.receptive_field = 0
        for i in range(n_layers):
            layers.append(nn.Sequential(
                Norm(norm_type=norm, in_channels=hidden_size),
                TemporalConvNet(input_channels=hidden_size,
                                hidden_channels=hidden_size,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                gated=gated,
                                activation=activation,
                                exponential_dilation=True,
                                n_layers=n_convs_layer,
                                causal_padding=True)
            )
            )
        self.convs = nn.ModuleList(layers)
        self.resnet = resnet
        activation_layer = get_layer_activation(activation=activation)

        self.readout = nn.Sequential(
            Lambda(lambda x: x[:, -readout_kernel_size:]),
            Rearrange('b s n c -> b n (c s)'),
            nn.Linear(hidden_size * readout_kernel_size, ff_size * horizon),
            activation_layer(),
            nn.Dropout(dropout),
            Rearrange('b n (c h) -> b h n c ', c=ff_size, h=horizon),
            nn.Linear(ff_size, output_size),
        )

        self.readout_1 = nn.Sequential(
            Rearrange('b 1 n 1 -> b n'),
            nn.Linear(8, 17),
            nn.ReLU(),
            Rearrange('b n -> b 1 n 1'),
        )

        self.window = readout_kernel_size
        self.horizon = horizon

    def forward(self, x, **kwargs):
        """"""
        # x: [b s n c]
        x = self.input_encoder(x)
        for conv in self.convs:
            x = x + conv(x) if self.resnet else conv(x)
        return self.readout_1(self.readout(x))

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[32])
        parser.opt_list('--ff-size', type=int, default=32, tunable=True, options=[256])
        parser.opt_list('--kernel-size', type=int, default=2, tunable=True, options=[2, 3])
        parser.opt_list('--n-layers', type=int, default=4, tunable=True, options=[2, 4, 6])
        parser.opt_list('--n-convs-layer', type=int, default=2, tunable=True, options=[1, 2])
        parser.opt_list('--dilation', type=int, default=2, tunable=True, options=[1, 2])
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.2])
        parser.opt_list('--gated', type=str_to_bool, tunable=False, nargs='?', const=True, default=False,
                        options=[True, False])
        parser.opt_list('--resnet', type=str_to_bool, tunable=False, nargs='?', const=True, default=True,
                        options=[True, False])
        parser.opt_list('--norm', type=str, default="batch", options=["none", "batch", "instance", "layer"])
        return parser


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


def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'cosine':
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = dict(eta_min=0.1 * args.lr, T_max=args.epochs)
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs


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
    parser.add_argument("--config", type=str, default='tcn.yaml')

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
    parser.add_argument('--grad-clip-val', type=float, default=.1)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)
    parser.add_argument('--monitor-metric', type=str, default='val_mae')

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    parser.add_argument("--adj-method", type=str, default='Pearson')

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

def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    ########################################
    # create logdir and save configuration #
    ########################################
    logger.info(args)
    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name,
                          args.model_name, exp_name)

    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp,
                  indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################
    dataset = PoreWaterPressure(root='./data', dataset_name=args.dataset_name)
    model_cls = get_model_classes(args.model_name)

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
            'x': 'data'  # predictors
        }

    connectivity = dataset.get_connectivity(method=args.adj_method,
                                            threshold=args.adj_threshold,
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
                                          horizon=args.horizon,
                                          window=args.window,
                                          delay=0,
                                          stride=args.stride)
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
        batch_size=args.batch_size,
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################
    loss_fn = MaskedMSE()

    scheduler_class, scheduler_kwargs = get_scheduler(args.lr_scheduler, args)

    metrics = {'mae': MaskedMAE(),
               'mape': MaskedMAPE(),
               'mse': MaskedMSE(),
               }

    additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                    input_size=dm.n_channels,
                                    output_size=dm.n_channels,
                                    window_size=dm.window)

    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # setup predictor
    predictor = Predictor(
        model_class=model_cls,              # our initialized model
        optim_class=torch.optim.Adam,  # specify optimizer to be used...
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},    # ...and parameters for its initialization
        model_kwargs=model_kwargs,
        loss_fn=loss_fn,               # which loss function to be used
        metrics=metrics,                # metrics to be logged during train/val/test
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
    )
    tb_logger = TensorBoardLogger(logdir, name="model")
    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir,
        save_top_k=1,
        monitor=args.monitor_metric,
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=tb_logger,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm='norm',
                         limit_train_batches=100,  # end an epoch after 100 updates
                         callbacks=[checkpoint_callback])

    trainer.fit(predictor, datamodule=dm)

    ########################################
    # testing                              #
    ########################################
    predictor.load_model(checkpoint_callback.best_model_path)
    predictor.freeze()

    trainer.test(predictor, datamodule=dm)

    output = trainer.predict(predictor, dataloaders=dm.test_dataloader())
    output = casting.numpy(output)
    y_hat, y_true, mask = output['y_hat'].squeeze(-1), \
                          output['y'].squeeze(-1), \
                          output['mask'].squeeze(-1)
    check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
    print(f'Test MAE: {check_mae:.2f}')
    return output

if __name__ == '__main__':
    args = parse_args()
    output = run_experiment(args)


    pass
