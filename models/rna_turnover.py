from met.hyperparameters import args
import numpy as np
import torch


class Model:
    """
    This is the mRNA turnover model.
    Cao, D., and Parker, R. (2001).
    Computational modeling of eukaryotic mRNA turnover. RNA 7, 1192–1212.
    https://doi.org/10.1017/ S1355838201010330.
    Suter, D.M., Molina, N., Gatfield, D., Schneider, K., Schibler, U., and Naef, F. (2011).
    Mammalian genes are transcribed with widely different bursting kinetics. Science 332, 472–474.
    https:// doi.org/10.1126/science. 1198817.

    Gu --> Gb, sb
    Gb --> Gu, su
    Gu --> Gu + A, 1
    A --> B, k2
    B --> BC1, r1
    BC1 --> BC2, r2
    BC2 --> BC3, r3
    BC3 --> BC4, r4
    BC4 --> BC5, r5
    BC5 --> C, r6
    C --> E, r7
    C --> D, k3
    E --> G, k8
    E --> F, k9
    D --> F, r7
    D --> L, k4
    G --> M, k10
    G --> null, k11
    F --> M, k8
    F --> I1, k4
    L --> I2, r8
    M --> null, k11
    M --> null, k4
    I1 --> null, k5
    I2 --> null, k5
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
        Gu, A, B, BC1, BC2, BC3, BC4, BC5, C, E, D, G, F, L, M, I1, I2 = population
        # propensities
        propensities[0] = Gu * k[0]
        propensities[1] = (1-Gu) * k[1]
        propensities[2] = Gu * k[2]
        propensities[3] = A * k[3]
        propensities[4] = B * k[4]
        propensities[5] = BC1 * k[5]
        propensities[6] = BC2 * k[6]
        propensities[7] = BC3 * k[7]
        propensities[8] = BC4 * k[8]
        propensities[9] = BC5 * k[9]
        propensities[10] = C * k[10]
        propensities[11] = C * k[11]
        propensities[12] = E * k[12]
        propensities[13] = E * k[13]
        propensities[14] = D * k[14]
        propensities[15] = D * k[15]
        propensities[16] = G * k[16]
        propensities[17] = G * k[17]
        propensities[18] = F * k[18]
        propensities[19] = F * k[19]
        propensities[20] = L * k[20]
        propensities[21] = M * k[21]
        propensities[22] = M * k[22]
        propensities[23] = I1 * k[23]
        propensities[24] = I2 * k[24]

    def reaction_matrix(self):
        # reaction matrix: species x reactions
        reaction_mat_left = torch.as_tensor([
            (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Gu
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # A
            (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # B
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC1
            (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC2
            (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC3
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC4
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC5
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # C
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # E
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # D
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),  # G
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),  # F
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # L
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),  # M
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),  # I1
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),  # I2
        ], dtype=self.default_dtype_torch, device=self.device)
        reaction_mat_right = torch.as_tensor([
            (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Gu
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # A
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # B
            (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC1
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC2
            (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC3
            (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC4
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # BC5
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # C
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # E
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # D
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # G
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # F
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # L
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),  # M
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),  # I1
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # I2
        ], dtype=self.default_dtype_torch, device=self.device)

        return reaction_mat_left, reaction_mat_right
    def initial_state(self):
        return self.initial_distribution

    def initial_time(self):
        return self.t0

    def rates(self):

        return np.array(self.default_rates)


# --------------------------------- model architecture -------------------------------------#
args.model = 'mRNA-Turnover'
args.num_species = 17  # Species number
args.num_reactions = 25  # reaction number
# Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation.
args.state_upper_bound = int(30)
args.init = [1] + [0 for _ in range(16)]  # Gu, A, B, BC1, BC2, BC3, BC4, BC5, C, E, D, G, F, L, M, I1, I2
args.initial_distribution = np.array(args.init).reshape(1, args.num_species)  # the parameter for the initial distribution
args.init_time = [0]
args.constrains = np.array([2] + [30 for _ in range(16)], dtype=int)
args.rate_parameters = (
    2.14,  # sb
    1.6,   # su
    20,    # rho
    10,    # k2
    1.69,  # r1
    2.86,  # r2
    1.51,  # r3
    6.2,   # r4
    3.33,  # r5
    4.61,  # r6
    1.3,   # r7
    2.1,   # k3
    1.91,  # k8
    1.63,  # k9
    1.3,   # r7
    23.38, # k4
    1.28,  # k10
    3,     # k11
    1.91,  # k8
    23.38, # k4
    1.5,   # r8
    3,     # k11
    23.38, # k4
    3.7,   # k5
    3.7,   # k5
)

# --------------------------------- gillespie simulation -------------------------------------#
args.end_time = 10
args.num_trajectories = 10000
args.cpu_cores = 10
args.record_time_slices = int(500)
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
args.cme_dt = 0.001  # Time step length of iterating the chemical master equation, depending on the reaction rates
args.max_step_initial_training = 5000  # Maximum number of training steps for the first time step (usually larger to ensure the accuracy)
args.max_step_training = 100  # Maximum number of steps of the rest time steps
args.total_cme_steps = int(args.end_time / args.cme_dt)  # Time step of iterating the chemical master equation, 36000 steps totally
args.clear_checkpoint = True
args.use_adaptive_cme_time = False  # adaptive time steps, use to increase speed, not use to increase accuracy, especially at beginning
args.adaptive_time_fold = 10  # adaptive folds of time steps
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
args.out_dir = "turnover/"
args.out_filename = None  # generate out dirs automatically
