import os
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')

group.add_argument(
    # '--Model'
    '--model',
    type=str,
    default='No',
    # choices=['No','cascade1','ToggleSwitch','cascade2','cascade3','repressilator','homo1','MM','AFL','GeneExp1',
    # 'GeneExp2', 'BirthDeath', 'Moran','Epidemic','EarlyLife2'],
    help='Models for master equation'
)

group.add_argument(
    # '--IniDistri'
    '--initial_distribution_type',
    type=str,
    default='delta',
    choices=['delta', 'poisson'],
    help='Initial Distribution type for the species, delta function or Poisson'
)

group.add_argument(
    # '--Tstep',
    "--total_cme_steps",
    type=int,
    default=1,
    help='Time step of iterating the dynamical equation P_tnew=T*P_t'
)

group.add_argument(
    # '--Para',
    "--model_type",
    type=float,
    default=1,
    help='specify model type, different types means different rate parameters or network topology'
)

group.add_argument(
    # '--loadTime',
    "--cme_checkpoint",
    type=int,
    default=1000,
    help='checkpoint of cme for continuous training when loading CMET'
)

group.add_argument(
    # '--loadTime',
    "--epoch_checkpoint",
    type=int,
    default=1000,
    help='checkpoint of cme for continuous training when loading CMET'
)

group.add_argument(
    # '--delta_t',
    "--cme_dt",
    type=float,
    default=0.05,
    help='Time step length of iterating the dynamical equation'
)

group.add_argument(
    # '--AdaptiveTFold',
    "--adaptive_time_fold",
    type=float,
    default=100,
    help='Explore the Adaptive T increase Fold'
)

group.add_argument(
    # '--boundary',
    "--boundary_condition_type",
    type=str,
    default='periodic',
    choices=['open', 'periodic'],
    help='type of boundary condition'
)

group.add_argument(
    # '--L',
    "--num_species",
    type=int,
    default=3,
    help='number of species'
)

group.add_argument(
    "--num_reactions",
    type=int,
    default=3,
    help='number of reactions'
)

group.add_argument(
    # '--Sites',
    "--num_spatial_sites",
    type=int,
    default=1,
    help='number of sites for spatial-extended systems'
)

group.add_argument(
    # '--order',
    "--species_order",
    type=int,
    default=1,
    help='forward or reverse order for species'
)

group.add_argument(
    # '--M',
    "--state_upper_bound",
    type=int,
    default=1e2,
    help='Upper limit of the Molecule number'
)

group.add_argument(
    '--beta',
    type=float,
    default=1,
    help='beta = 1 / k_B T'
)

group.add_argument(
    # '--Percent',
    "--epoch_percent",
    type=float,
    default=0.1,
    help='Default 0.1; Percent: 1. Percent of laster epochs used to calculate free energy; 2. Percent of epochs to '
         'reduce lr'
)

group = parser.add_argument_group('network parameters')

group.add_argument(
    # '--net',
    "--network_type",
    type=str,
    default='rnn',
    choices=['rnn', 'transformer'],
    help='network type for training'
)

group.add_argument(
    '--net_depth',
    type=int,
    default=3,
    help='network depth'
)

group.add_argument(
    '--net_width',
    type=int,
    default=64,
    help='network width'
)

group.add_argument(
    # '--d_model',
    "--embedding_dimension",
    type=int,
    default=64,
    help='transformer'
)

group.add_argument(
    # '--d_ff',
    "--feed_forward_dimension",
    type=int,
    default=128,
    help='transformer'
)

group.add_argument(
    # '--n_layers',
    "--num_encoder_layers",
    type=int,
    default=2,
    help='transformer'
)

group.add_argument(
    '--n_head',
    type=int,
    default=2,
    help='transformer'
)

group.add_argument(
    '--half_kernel_size',
    type=int,
    default=1,
    help='(kernel_size - 1) // 2'
)

group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype'
)

group.add_argument(
    '--bias',
    action='store_true',
    help='use bias'
)

group.add_argument(
    # '--AdaptiveT',
    "--use_adaptive_cme_time",
    action='store_true',
    help='use AdaptiveT'
)

group.add_argument(
    '--binary',
    action='store_true',
    help='use binary conversion'
)

group.add_argument(
    '--lr_schedule_type',
    type=int,
    default=1,
    help='lr rate schedulers'
)

group.add_argument(
    # '--lossType',
    "--loss_type",
    type=str,
    default='kl',
    choices=['l2', 'kl', 'he', 'ss'],
    help='Loss functions: l2, KL-divergence, and Hellinger'
)

group.add_argument(
    '--conservation',
    type=int,
    default=1,
    help='imposing conservation of some quantities'
)

group.add_argument(
    '--reverse',
    action='store_true',
    help='with reverse conditional probability'
)

group.add_argument(
    '--loadCMET',
    action='store_true',
    help='load CMET at later time points'
)

group.add_argument(
    '--res_block',
    action='store_true',
    help='use res block'
)

group.add_argument(
    '--epsilon',
    type=float,
    default=0,  # default=1e-39,
    help='small number to avoid 0 in division and log'
)

group.add_argument(
    # '--initialD',
    "--initial_distribution",
    type=float,
    default=1,  # default=1e-39,
    help='the parameter for the initial distribution'
)

group.add_argument(
    # '--MConstrain',
    "--constrain_states",
    type=int,
    default=np.zeros(1, dtype=int),  # default=1e-39,
    help='MConstrain'
)

group = parser.add_argument_group('optimizer parameters')

group.add_argument(
    '--seed',
    type=int,
    default=0,
    help='random seed, 0 for randomized'
)

group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer'
)

group.add_argument(
    '--batch_size',
    type=int,
    default=10 ** 3,
    help='number of samples'
)

group.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='learning rate'
)

group.add_argument(
    '--max_step',
    type=int,
    default=10 ** 3,
    help='maximum number of steps'
)

group.add_argument(
    # '--max_stepAll',
    "--max_step_initial_training",
    type=int,
    default=10 ** 4,
    help='maximum number of steps for first point training, usually costs more steps than later training'
)

group.add_argument(
    # '--max_stepLater',
    "--max_step_training",
    type=int,
    default=50,
    help='maximum number of steps of later time step'
)

group.add_argument(
    '--lr_schedule',
    action='store_true',
    help='use learning rate scheduling'
)

group.add_argument(
    '--beta_anneal',
    type=float,
    default=0,
    help='speed to change beta from 0 to final value, 0 for disabled'
)

group.add_argument(
    '--clip_grad',
    type=float,
    default=0,
    help='global norm to clip gradients, 0 for disabled'
)

group = parser.add_argument_group('system parameters')

group.add_argument(
    "--rate_parameters",
    type=tuple,
    default=(0, 0, 0, 0),
    help="rate parameters"
)

group.add_argument(
    "--sim_end_time",
    type=float,
    default=1000,
    help="end time of simulation"
)

group.add_argument(
    "--num_trajectories",
    type=int,
    default=1000,
    help="total number of trajectories generated from simulation"
)

group.add_argument(
    "--cpu_cores",
    type=int,
    default=4,
    help="number of cpu cores used for simulation"
)

group.add_argument(
    "--record_time_slices",
    type=int,
    default=100,
    help="number of time slices recorded from simulation"
)

group.add_argument(
    "--run_sim",
    action="store_true",
    help="run simulation, otherwise stored npz file will be loaded"
)

group.add_argument(
    "--save_sim",
    action="store_true",
    help="save simulation results in npz file"
)

group.add_argument(
    "--save_sim_file",
    type=str,
    default="sim.npz",
    help="file name for saving simulation results"
)

group = parser.add_argument_group('system parameters')

group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance'
)

group.add_argument(
    '--clear_checkpoint',
    action='store_true',
    help='clear checkpoint'
)

group.add_argument(
    '--print_step',
    type=int,
    default=100,
    help='number of steps to print log, 0 for disabled'
)

group.add_argument(
    '--save_step',
    type=int,
    default=100,
    help='number of steps to save network weights, 0 for disabled'
)

group.add_argument(
    '--visual_step',
    type=int,
    default=100,
    help='number of steps to visualize samples, 0 for disabled'
)

group.add_argument(
    '--save_sample',
    action='store_true',
    help='save samples on print_step'
)

group.add_argument(
    '--print_sample',
    type=int,
    default=1,
    help='number of samples to print to log on visual_step, 0 for disabled'
)

group.add_argument(
    '--print_grad',
    action='store_true',
    help='print summary of gradients for each parameter on visual_step'
)

group.add_argument(
    '--cuda',
    type=int,
    default=-1,
    help='ID of GPU to use, -1 for disabled'
)

group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs'
)

group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled'
)

group.add_argument(
    '--saving_data_time_step',
    type=list,
    default=[0, 1e2, 5e2, 2e3, 1e4, 2e4, 5e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6],
    help='To save data at which time steps (give in a list)'
)

group.add_argument(
    '--training_loss_print_step',
    type=list,
    default=[0, 1, 2, 101, 1001, 2e3, 1e4, 1e5, 2e5, 3e5, 4e5, 5e5],
    help='To print training loss at which time steps (give in a list)'
)

args = parser.parse_args(args=["--cuda", 0, "--dtype", "float32"])


# --------------------------------- cuda and env settings -------------------------------------#
np.seterr(all='raise')
np.seterr(under='warn')
np.set_printoptions(precision=8, linewidth=160)

args.cuda = 0  # use the first GPU, -1 for using CPU
args.dtype = 'float64'  # use float64 if cpu or cuda V-100

args.device = torch.device('cpu' if args.cuda < 0 else 'cuda:0')

if args.dtype == 'float32':
    default_dtype = np.float32
    default_dtype_torch = torch.float32
elif args.dtype == 'float64':
    default_dtype = np.float64
    default_dtype_torch = torch.float64
else:
    raise ValueError('Unknown dtype: {}'.format(args.dtype))

args.default_dtype_torch = default_dtype_torch


torch.set_default_dtype(default_dtype_torch)
torch.set_printoptions(precision=8, linewidth=160)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

args.seed = 0  # 0 for using random seed, 1337 is used for nanoGPT training

if not args.seed:
    args.seed = np.random.randint(1, 10 ** 8)
np.random.seed(args.seed)
torch.manual_seed(args.seed)



def print_args_value(input_args):
    for i in list(vars(input_args).keys()):
        print(f"{i}: {vars(input_args)[i]}")


if __name__ == "__main__":
    print_args_value(args)
