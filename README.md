# MET
This is the software package for the manuscript titled "Language Models as Master Equation Solvers".

MET is the abbreviation for Master Equation Transformer. It utilizes the language model as a solver for a well-defined master equation. The trained network can accept prompts containing rate parameters, initial distributions, and time values as inputs, and the output is the joint distribution of the system state. This approach allows the trained network to provide the most general solution of the Kolmogorov forward equation. MET can also be expanded to include systems in physical, chemical, biological, and financial domains by providing a valid set of training models.

The `models` folder contains example models that can be used to reproduce the results mentioned in the manuscript. 


System requirements: 

Run the codes requires Python >= 3.8 and PyTorch >= 1.10.


# Model

To apply the package to a new model, you need to provide a `.py` file that contains all the necessary information.

The `.py` file should include a Python class called `Model` which holds the model parameters, matrix masks, propensity functions, reaction matrices, initial state, initial time, and rate parameters.

Furthermore, the `.py` file should also include the parameters of the model architecture, the hyperparameters for generating the reward model set, and the hyperparameters for training MET.

Please refer to the existing examples in the `models` folder.

# Training
Once the `model.py` file has been provided, the reward model set can be generated and the MET can be trained by simply running the scripts within `main.py` on a machine equipped with a CUDA device.

The scripts will first generate the reward model set, following the hyperparameters specified in the `model.py` file. The trained reward models will then be stored, and subsequently used to train the MET.

The time required to generate the reward model set depends on the number of dimensions in the state space of the master equation, as well as the total number of time steps required to reach the observation end time. 
Typically, generating a reward model set for a 4-species model with $10^4$ time steps will take approximately 10 hours.

The time needed to train the MET depends on the hyperparameters of the network. For a network with 0.4 million parameters, training it for 5,000 epochs will take around 200 hours. 

# State sampling 
When the training of MET is finished, the scripts will store the final network weights for further inference tasks. Several example Python scripts are presented in the 'sample' folder.

The file `met_autoreg_bc.py` was used to calculate the bimodality coefficients of the marginal distributions for the autoregulation model.

The file `met_autoreg_inference.py` was used to infer the rate parameters for a provided trajectory samples for the autoregulation model.

The files `rnn_sample.py` and `met_sample.py` provide general scripts for sampling states based on the reward model set and the trained MET network.

The file `traj_sample.py` was used to reproduce the trajectory ensemble samples for the birth-death model.

It should be noted that the prompts for inference should be in the same format as the training process. We have used a sequential format, i.e., logarithm of rate parameters -> initial states -> time value, for training and inference. However, other formats are also possible.

# Verification of the results
We utilized the stochastic simulation algorithm developed by Gillespie to validate the results of MET and the reward models.

The simulation script can be found in the `met/gillespie.py` file. This script relies on the [biocircuits](https://pypi.org/project/biocircuits/) package.

To run the simulation, you need to provide a transition matrix along with its corresponding propensity function. Once you have specified the initial population, time points for recording, number of trajectories, and number of CPU threads in the `model.py` file, you can easily execute the simulation using the script in `ssa.py`.

# Plot the results
Example scripts for reproducing the figures in the manuscripts are provided in the file `toggle_plot.ipynb`.

The file can be opened with applications such as vscode, jupyter notebook, or pycharm.

Running the code requires state sampling files from Gillespie, RNN, and MET, as mentioned in state sampling.

Time-dependent averaged trajectories, marginal distributions, pair-wise joint distributions, and comparison of means are used to visualize the sampling results.