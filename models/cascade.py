from met.hyperparameters import args
import numpy as np
import torch


class Model:
    """
    This is the intracellular signaling cascade model.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.default_dtype_torch = kwargs["default_dtype_torch"]
        self.device = kwargs["device"]
        self.N = kwargs["num_species"]
        self.M = kwargs["num_reactions"]
        self.BS = kwargs["batch_size"]
        self.U = kwargs["state_upper_bound"]
        self.init = kwargs["init"]
        self.init = torch.as_tensor(self.init, dtype=self.default_dtype_torch, device=self.device)
        self.initial_distribution = torch.as_tensor(kwargs["initial_distribution"], dtype=self.default_dtype_torch, device=self.device)
        self.t0 = torch.as_tensor(kwargs["init_time"], dtype=self.default_dtype_torch, device=self.device)
        self.constrains = kwargs["constrains"]
        self.default_rates = np.array(kwargs["rate_parameters"])
        self.species_order = kwargs["species_order"]

    def propensity_function(self, Win, Wout, cc, SampleNeighbor1D_Win, SampleNeighbor1D, NotHappen_in_low, NotHappen_in_up,
                            NotHappen_out_low, NotHappen_out_up):
        Propensity_in = torch.prod(Win, 1) * cc  # torch.tensor(r, dtype=torch.float64).to(args.device)
        Propensity_out = torch.prod(Wout, 1) * cc  # torch.tensor(r, dtype=torch.float64).to(args.device)

        return Propensity_in, Propensity_out

    @staticmethod
    def sim_propensity_function(
            propensities: np.ndarray,
            population: np.ndarray,
            t: float,
            *rates
    ):
        propensities[0] = rates[0]
        for i in range(15):
            propensities[2*i+1] = population[i] * rates[2*i+1]
        for i in range(1, 15):
            propensities[2*i] = population[i-1] * rates[2*i]

    def reaction_matrix(self):
        reaction_mat_left = torch.zeros((self.N, 2 * self.N)).to(self.device)
        for i in range(self.N):  # decay
            reaction_mat_left[i, 2 * i + 1] = 1
        for i in range(1, self.N):  # decay
            reaction_mat_left[i - 1, 2 * i] = 1
        reaction_mat_right = torch.zeros((self.N, 2 * self.N)).to(self.device)
        reaction_mat_right[0, 0] = 1
        for i in range(1, self.N):  # decay
            reaction_mat_right[i, 2 * i] = 1

        return reaction_mat_left, reaction_mat_right

    def initial_state(self):
        return self.initial_distribution

    def initial_time(self):
        return self.t0

    def rates(self):
        beta = 10
        k = 5
        gamma = 1
        r = torch.zeros(2 * self.N)
        r[0] = beta
        for ii in range(self.N):  # decay
            r[2 * ii + 1] = gamma
        for ii in range(1, self.N):  # decay
            r[2 * ii] = k
        return r


# --------------------------------- model architecture -------------------------------------#
args.model = 'Cascade'
args.num_species = 15  # Species number
args.num_reactions = 30  # reaction number
# Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation.
args.state_upper_bound = int(10)
args.init = [0 for _ in range(15)]
args.initial_distribution = np.array(args.init).reshape(1, args.num_species)  # the parameter for the initial distribution
args.init_time = [0]
args.constrains = np.array([0], dtype=int)
args.rate_parameters = [10 for i in range(30)]  # sx, sy, dx, dy, by, bx, uy, ux
args.rate_parameters[1::2] = [1 for i in range(15)]
args.rate_parameters[2::2] = [5 for i in range(14)]
args.rate_parameters = tuple(args.rate_parameters)

# --------------------------------- gillespie simulation -------------------------------------#
args.end_time = 10
args.num_trajectories = 10000
args.cpu_cores = 10
args.record_time_slices = int(10/0.01)
args.run_sim = True
args.save_sim = True
args.save_sim_file = None

# --------------------------------- generate van dataset -------------------------------------#
args.net = 'rnn'  # name of the reward model, could also be transformer
args.net_depth = 1  # including output layer and not input data layer for rnn
args.net_width = 32  # number of neurons for rnn, 32 is better than 16
args.loss_type = 'kl'  # use kl-divergence for reward loss
args.initial_distribution_type = "delta"
args.van_batch_size = 1000  # Number of batch samples
args.cme_dt = 0.01  # Time step length of iterating the chemical master equation, depending on the reaction rates
args.max_step_initial_training = 5000  # Maximum number of training steps for the first time step (usually larger to ensure the accuracy)
args.max_step_training = 100  # Maximum number of steps of the rest time steps
args.total_cme_steps = int(args.end_time / args.cme_dt)  # Time step of iterating the chemical master equation, 36000 steps totally
args.clear_checkpoint = True
args.use_adaptive_cme_time = False  # adaptive time steps, use to increase speed, not use to increase accuracy, especially at beginning
args.adaptive_time_fold = 5  # adaptive folds of time steps
args.training_loss_print_step = 10  # print every 10 cme steps, overall 3600 prints
args.saving_data_time_step = args.training_loss_print_step  # save data and net state every 10 cme steps
args.bits = 1
args.binary = False  # binary conversion
if args.binary:
    args.bits = int(np.ceil(np.log2(args.state_upper_bound)))
args.print_step = 10  # print output every 10 cme steps
args.save_step = 1  # save reward net state and samples every 10 cme steps
args.epoch_percent = 0.2
args.clip_grad = 1
args.epsilon = 1e-30  # solving divide by zero problem
args.lr_schedule = False  # True
args.van_bias = True  # True for training reward model
args.van_lr = 0.001

# --------------------------------- train met -------------------------------------#
args.variable_dimension = 64  # dimensionality of the variable matrix including prompt and states
args.embedding_dimension = 64  # transformer emb_dim
args.feed_forward_dimension = 1024  # transformer ff_dim
args.num_encoder_layers = 8  # transformer n_layer
args.n_head = 16  # transformer n_head
args.block_size = 128  # maximum input length for met
args.lr = 0.001  # initial learning rate
args.met_batch_size = 100  # Number of batch samples
args.bias = False  # False for training met
args.dropout_rate = 0.0  # for met-gpt
args.weight_decay = 1e-1  # for configure met-gpt optimizer
args.beta1 = 0.9  # for configure met-gpt optimizer
args.beta2 = 0.999  # for configure met-gpt optimizer
args.decay_lr = False  # whether to decay the learning rate
args.epochs = 10000  # usually should be 5000-10000 epochs for convergent training
args.load_met = False
args.start_epoch = 0  # changed when loading pretrain met state file
args.last_epoch = 4999  # specify last epoch for loading met pretrain state file


# --------------------------------- output directory -------------------------------------#
args.out_dir = "cascade/"
args.out_filename = None  # generate out dirs automatically
