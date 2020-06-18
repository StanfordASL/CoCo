# CoCo: Combinatorial Offline, Convex Online
This repository contains code for [Learning Mixed-Integer Convex Optimization Strategies for Robot Planning and Control](https://arxiv.org/abs/2004.03736) by Abhishek Cauligi\*, Preston Culbertson\*, Bartolomeo Stellato, Dimitris Bertsimas, Mac Schwager, and Marco Pavone (\* denotes equal contribution).

## Installation ##
This repository uses cvxpy for the construction of the optimization problems and PyTorch for training the neural network models. The necessary Python packages can be installed by running the following script.
```
pip3 install -r requirements_cython.txt
pip3 install -r requirements.txt
```
We also use the [Gurobi][https://support.gurobi.com/hc/en-us/community/posts/360046430451/comments/360005981732] and [Mosek][https://www.mosek.com/downloads/] commercial solvers for solving our problems.

Further, define an environment variable `MLOPT` that points to the working directory where this packagee is installed.

## Useage ##
This repo contains examples for three systems in robot planning and control:
1. Cart-pole with contact
<p align="center"><img width="35%" src="img/cart-pole.png"/></p>

2. Free-flyer motion planning
<p align="center"><img width="35%" src="img/free-flyer.png"/></p>

3. Dexterous grasping
<p align="center"><img width="35%" src="img/dexterous_manipulation.png"/></p>

The MICP for each problem is defined in `{system}/problem.jl` using either the [JuMP](https://github.com/JuliaOpt/JuMP.jl) or [Convex](https://github.com/JuliaOpt/Convex.jl) modeling framework. The `{system}/data_collection.ipynb` notebook must be run first to generate the MICP datasets later used in training.

Each system has a `{system}.ipynb` notebook that steps through the strategy construction, classifier training, and evaluation of the trained strategy predictions. The neural network models are trained separately using Python scripts found in the `pytorch` directory and the strategy and network parameters are saved in the `models` directory.


## Quick Start ##
An example notebook can be run through:
```
jupyter notebook cartpole.ipynb 
```
