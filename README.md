# Differentiable Invariant Causal Discovery

This is the implementation of the paper **Differentiable Invariant Causal Discovery**

## Requirements
+ torch == 1.9.0+cu111
+ Numpy
+ python3

## Datasets

We have put the example data of 10 nodes and 40 edges under linear setting in the folder `data`. The full dataset could be downloaded [here](https://drive.google.com/drive/folders/1Bihhqqu1bEHzNcb-ZKJG_xw25vo7C77q?usp=sharing). 

# Commands

You can run NOTEARS and DICD for the linear experiments for ER4 graph with 10 nodes via the following codes:
```
python linear_exp.py --s0 40 --d 10 --method NOTEARS --graph_type ER
python linear_exp.py --s0 40 --d 10 --method DICD --graph_type ER
```

You can run NOTEARS and DICD for the nonlinear experiments for SF4 graph with 10 nodes via the following codes:
```
python nonlinear_exp.py --s0 40 --d 10 --method NOTEARS --graph_type SF
python nonlinear_exp.py --s0 40 --d 10 --method DICD --graph_type SF
```

For DAG-GNN and DAG-NoCurl, We follow the official implementations with the link provided as follows:
+ DAG-GNN: https://github.com/fishmoon1234/DAG-GNN
+ DAG-NoCurl: https://github.com/fishmoon1234/DAG-NoCurl

the scripts are as follows:
```
python linear_exp.py --s0 40 --d 10 --method DAG-GNN --graph_type ER
python linear_exp.py --s0 40 --d 10 --method NoCurl --graph_type ER
python linear_exp.py --s0 40 --d 10 --method DARING --graph_type ER
python non_linear_exp.py --s0 40 --d 10 --method DAG-GNN --graph_type ER
python non_linear_exp.py --s0 40 --d 10 --method NoCurl --graph_type ER
python non_linear_exp.py --s0 40 --d 10 --method DARING --graph_type ER
```





