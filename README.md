# MET
This is the software package for the manuscript "Language models as master equation solvers".

MET is the short name for Master Equation Transformer. 
It repurposes the language model as a solver for a well-defined master equation. 
The trained network can accept prompts that contain rate parameters, initial distributions and time values as inputs, and the outputs is the joint distribution of the system state. 
With this approach, the trained network can provide the most general solution of the Kolmogorov forward equation. 
MET can also be extended to systems in physical, chemical, biological and financial systems by providing the valid set of training models. 

Example models to reproduce the results in the manuscript are contained in the `models` folder. 


System requirements: 

Run the codes requires Python >= 3.8 and PyTorch >= 1.10.


# Model

For applying the package to new model, a `.py` file which contains all the necessary information should be provided. 

Within the `.py` file, a python class named `Model` is used to contain the model parameters, matrix masks, propensity functions, reaction matrices, initial state, initial time, and rate parameters. 

Additionally, the parameters of the model architecture, the hyperparameters for generating the reward model set, and the hyperparameters for training MET should also be included in the `.py` file.

Please refer to the existing examples in the `models` folder.

# Training
Once the `model.py` file is provided, generating of the reward model set and training of MET can be done by simply run the scripts within `main.py`. 

The scripts will first generate the reward model set as instructed by the hyperparameters contained in the `model.py` file, store the trained reward models, and then use the stored reward models to train MET. 

The time cost for generating the reward model set depends on the number of dimensions of the state space of the master equation, and also on the total time steps for reaching the end time of observation. 
Typically, for generating a reward model set for a 4-species model with $10^4$ time steps will cost about 10 hours.

The time cost for training MET depends on the hyperparameters of the network, for a network with 

# Explanation of hyperparameters

- `args.cuda`: specify the index of which CUDA device is used for training. Currently, we do not support distributive training.
- `args.dtype`: data type of the tensor, could be `float32` or `float64` when Tesla-V100 is used for training.
- `args.seed`: random seed for training initialization.
- `args.num_species`: number of state space dimensions.
- `args.num_reactions`: number of accessible jumps. 
- `args.state_upper_bound`: number of possible states within one state space dimension.