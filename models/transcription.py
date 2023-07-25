from met.hyperparameters import args
import numpy as np
import torch

class Model:
    """
    This is the transcription model which account for gene stochastic switch.

    Gu --> Gb, sb
    Gb --> Gu, su
    Gu --> Gu + A, 1
    A --> null, k2
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

    # It is used to constrain the count of certain species.
    # For example, DNA typically only has the count of 0 or 1 inside a cell.
    # The "Mask" function allows only the reactions with such a proper count to occur.
    def matrix_mask(self, SampleNeighbor1D_Win, WinProd):
        mask = torch.ones_like(WinProd)
        Gu = SampleNeighbor1D_Win[:, 0, :]  # Gu for different reactions: the second dimension of SampleNeighbor1D_Win is the label of species
        Gu1 = 1 - Gu
        mask[Gu[:, 0] != 1, 0] = 0  # The second dimension of Gu and mask is the label of reactions
        mask[Gu[:, 2] != 1, 2] = 0
        mask[Gu1[:, 1] != 1, 1] = 0

        return mask

    def propensity_function(self, Win, Wout, cc, SampleNeighbor1D_Win, SampleNeighbor1D, NotHappen_in_low, NotHappen_in_up,
                            NotHappen_out_low, NotHappen_out_up):

        WinProd = torch.prod(Win, 1)
        mask = self.matrix_mask(SampleNeighbor1D_Win, WinProd)
        Propensity_in = WinProd * mask * cc
        WoutProd = torch.prod(Wout, 1)
        mask = self.matrix_mask(SampleNeighbor1D, WoutProd)
        Propensity_out = WoutProd * mask * cc

        return Propensity_in, Propensity_out

    @staticmethod
    def sim_propensity_function(
            propensities: np.ndarray,
            population: np.ndarray,
            t: float,
            *k
    ):
        # species
        Gu, A, = population
        # propensities
        propensities[0] = Gu * k[0]
        propensities[1] = (1-Gu) * k[1]
        propensities[2] = Gu * k[2]
        propensities[3] = A * k[3]

    def reaction_matrix(self):
        # reaction matrix: species x reactions
        reaction_mat_left = torch.as_tensor([
            (1, 0, 1, 0),  # Gu
            (0, 0, 0, 1),  # A
        ], dtype=self.default_dtype_torch, device=self.device)
        reaction_mat_right = torch.as_tensor([
            (0, 1, 1, 0),  # Gu
            (0, 0, 1, 0),  # A
        ], dtype=self.default_dtype_torch, device=self.device)

        return reaction_mat_left, reaction_mat_right
    def initial_state(self):
        return self.initial_distribution

    def initial_time(self):
        return self.t0

    def rates(self):
        return np.array(self.default_rates)


# --------------------------------- model architecture -------------------------------------#
args.model = 'Transcription'
args.num_species = 2  # Species number
args.num_reactions = 4  # reaction number
# Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation.
args.state_upper_bound = int(10)
args.init = [1, 0]
args.initial_distribution = np.array(args.init).reshape(1, args.num_species)  # the parameter for the initial distribution
args.init_time = [0]
args.constrains = np.array([2, 10], dtype=int)
args.rate_parameters = (
    0.0214,
    0.016,
    1,
    0.603,
)

# --------------------------------- gillespie simulation -------------------------------------#
args.end_time = 200
args.num_trajectories = 10000
args.cpu_cores = 10
args.record_time_slices = int(200)
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
args.cme_dt = 0.0001  # Time step length of iterating the chemical master equation, depending on the reaction rates
args.max_step_initial_training = 5000  # Maximum number of training steps for the first time step (usually larger to ensure the accuracy)
args.max_step_training = 100  # Maximum number of steps of the rest time steps
args.total_cme_steps = int(args.end_time / args.cme_dt)  # Time step of iterating the chemical master equation, 36000 steps totally
args.clear_checkpoint = True
args.use_adaptive_cme_time = True  # adaptive time steps, use to increase speed, not use to increase accuracy, especially at beginning
args.adaptive_time_fold = 100  # adaptive folds of time steps
args.training_loss_print_step = 10  # print every 10 cme steps, overall 3600 prints
args.saving_data_time_step = args.training_loss_print_step  # save data and net state every 10 cme steps
args.bits = 1
args.binary = False  # binary conversion
if args.binary:
    args.bits = int(np.ceil(np.log2(args.state_upper_bound)))
args.print_step = 10  # print output every 10 cme steps
args.save_step = 100  # save reward net state and samples every 10 cme steps
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

# batch_size for V-100 32G memory is 100

# --------------------------------- output directory -------------------------------------#
args.out_dir = "transcription/"
args.out_filename = None  # generate out dirs automatically
