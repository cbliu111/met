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
Once the `model.py` file is provided, generating of the reward model set and training of MET can be done by simply run the scripts within `main.py` on a machine equipped with CUDA device. 

The scripts will first generate the reward model set as instructed by the hyperparameters contained in the `model.py` file, store the trained reward models, and then use the stored reward models to train MET. 

The time cost for generating the reward model set depends on the number of dimensions of the state space of the master equation, and also on the total time steps for reaching the end time of observation. 
Typically, for generating a reward model set for a 4-species model with $10^4$ time steps will cost about 10 hours.

The time cost for training MET depends on the hyperparameters of the network, for a network with 0.4 M parameters, training of 5,000 epochs takes about 200 hours. 

# State sampling 
When the training of MET is finished, the scripts will store the final network weights for further inference tasks. 
Several example Python scripts are presented in the `sample` folder. 

The file `met_autoreg_bc.py` was used to calculate the bimodulity coefficients of the marginal distributions for the autoregulation model. 

The file `met_autoreg_inference.py` was used to inference the rate parameters for a provided trajectory samples for the autoregulation model. 

File `met_sample.py` and `rnn_sample.py` provides a general scripts for sampling states based on the reward model set and the trained MET network. 

File `traj_sample.py` was used to reproduce the trajectory ensemble samples for the birth-death model.

It should be noted that the prompts for inferencing should be the same format as the training process.
We have used a sequential format, i.e. logarithm of rate parameters -> initial states -> time value, for training and inferencing. 
However, other formats are also possible. 

# Verification of the results
We used the stochastic simulation algorithm by Gillespie for the verification of the results of MET and the reward models. 

The simulation script is included in the `met/gillespie.py` file. 
The script is based on the [biocircuits](https://pypi.org/project/biocircuits/) package. 

For running the simulation, a transition matrix with corresponding propensity function should be provided. 
After specifying the initial population, time points for recording, number of trajectories and number of CPU threads in the `model.py` file, the simulation can be simply done using the script in `ssa.py`.

# Plot the results
Example scripts from reproducing the figures in the manuscripts are provided in the file `toggle_plot.ipynb`. 

The file can be opened with applications such as vscode, jupyter notebook or pycharm. 

Running the code requires state sampling files from Gillespie, RNN and MET. 

Time-dependent averaged trajectories, marginal distributions, pair-wise joint distributions and comparison of mean are used to visualize the sampling results. 