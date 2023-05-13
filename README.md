# Spatiotemporal-Prediction
# 黏土心墙坝渗流监测数据的深度学习方法研究

This repository contains the code for the reproducibility of the experiments presented in the dissertation "Research on Deep Learning Model for Seepage Safety Monitoring of Clay-core Wall Dams". 
**Authors**: Pan Liao, Xiaoqing Li


## Installation

We provide a requirements list with all the project dependencies in `requirements.txt`. 

## Configuration files

The `config/` directory stores all the configuration files used to run the experiment. `config/` stores model configurations used for experiments on imputation.

## Experiments

The scripts used for the experiment in the paper are in the `run_experiment.py` and `run_inference.py`.

* `run_experiment.py` is used to compute the metrics for the deep pore water pressure prediction methods. An example of usage for GAT-TCN model is

	```bash
	python run_experiment.py --config gat_tcn.yaml --model-name gat_tcn --dataset-name grin
	```

* `run_inference.py` is used for the experiments on sparse datasets using pre-trained models. An example of usage is

	```bash
	python run_inference.py --config inference.yaml --model-name gat_tcn --dataset-name grin --exp-name 20230509T123018_21380673
	```

 
