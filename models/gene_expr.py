from met.hyperparameters import args
import numpy as np
import torch

class Model:
    """
    This is the gene expression model without regulation.
    DNA --> mRNA, kr
    mRNA --> mRNA + protein, kp
    mRNA --> null, yr
    protein --> null, yp
    """
    def __init__(self, **kwargs):
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
        self.default_rates = kwargs["rate_parameters"]
        self.species_order = kwargs["species_order"]

    def propensity_function(self, Win, Wout, cc, SampleNeighbor1D_Win, SampleNeighbor1D, NotHappen_in_low, NotHappen_in_up,
                            NotHappen_out_low, NotHappen_out_up):
        Propensity_in = torch.prod(Win, 1) * cc
        Propensity_out = torch.prod(Wout, 1) * cc

        return Propensity_in, Propensity_out

    @staticmethod
    def sim_propensity_function(
            propensities: np.ndarray,
            population: np.ndarray,
            t: float,
            kr: float,
            kp: float,
            yr: float,
            yp: float,
    ):
        # species
        r, p = population
        # propensities
        propensities[0] = kr
        propensities[1] = kp * r
        propensities[2] = yr * r
        propensities[3] = yp * p

    def reaction_matrix(self):
        assert self.species_order == 1 or self.species_order == 2, "species order can only be 1 or 2"

        # reaction matrix: species x reactions
        reaction_mat_left = torch.as_tensor([
            (0, 1, 1, 0),
            (0, 0, 0, 1)
        ], dtype=self.default_dtype_torch, device=self.device)
        reaction_mat_right = torch.as_tensor([
            (1, 1, 0, 0),
            (0, 1, 0, 0)
        ], dtype=self.default_dtype_torch, device=self.device)

        # inverse the order of mRNA and protein, Scenario 2, should not make significant difference
        if self.species_order == 2:
            reaction_mat_left = torch.as_tensor([
                (0, 0, 0, 1),
                (0, 1, 1, 0)
            ], dtype=self.default_dtype_torch, device=self.device)
            reaction_mat_right = torch.as_tensor([
                (0, 1, 0, 0),
                (1, 1, 0, 0)
            ], dtype=self.default_dtype_torch, device=self.device)

        return reaction_mat_left, reaction_mat_right

    def initial_state(self):
        return self.initial_distribution

    def initial_time(self):
        return self.t0

    def rates(self):
        rates = np.array(self.default_rates)

        return rates


# --------------------------------- model architecture -------------------------------------#
args.model = 'GeneExp'  # model name
args.num_species = 2  # Species number, N
args.num_reactions = 4  # reaction number, M
# Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation.
args.state_upper_bound = int(100)  # U
args.init = [0, 0]  # initial state
args.initial_distribution = np.array(args.init).reshape(1, args.num_species)  # the parameter for the initial distribution
args.init_time = [0]
args.constrains = np.zeros(1, dtype=int)  # setting as [0] means setting all species in constrains [0, M]
# used to check if different dependency order of conditional probability makes a different
# 1: mRNA, protein
# 2: protein, mRNA
args.species_order = 1
args.model_type = 2  # used to select model based on parameters

if args.model_type == 1:
    args.rate_parameters = (0.01, 1, 0.1, 0.002)
elif args.model_type == 2:
    args.rate_parameters = (0.1, 0.1, 0.1, 0.002)


# --------------------------------- gillespie simulation -------------------------------------#
args.end_time = 3600  # 1 hr
args.num_trajectories = 10000  # number of trajectories sampled, usually set for 1e4 or 1e5
args.cpu_cores = 10  # cpu cores used for parallel simulation
args.record_time_slices = 1000  # how may time points is recorded? np.linspace(0, end_time, record_time_slices)
args.run_sim = True
args.save_sim = True
args.save_sim_file = None


# --------------------------------- generate van dataset -------------------------------------#
args.net = 'rnn'  # name of the reward model, could also be transformer
args.net_depth = 1  # including output layer and not input data layer for rnn
args.net_width = 128  # number of neurons for rnn, 32 is better than 16
args.loss_type = 'kl'  # use kl-divergence for reward loss
args.initial_distribution_type = "delta"
args.van_batch_size = 1000  # Number of batch samples
args.cme_dt = 0.1  # Time step length of iterating the chemical master equation, depending on the reaction rates
args.max_step_initial_training = 10000  # Maximum number of training steps for the first time step (usually larger to ensure the accuracy)
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
args.save_step = 10  # save reward net state and samples every 10 cme steps
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
args.n_head = 8  # transformer n_head
args.block_size = 128  # maximum input length for met
args.lr = 0.001  # initial learning rate
args.met_batch_size = 1000  # Number of batch samples
args.bias = False  # False for training met
args.dropout_rate = 0.0  # for met-gpt
args.weight_decay = 1e-1  # for configure met-gpt optimizer
args.beta1 = 0.9  # for configure met-gpt optimizer
args.beta2 = 0.999  # for configure met-gpt optimizer
args.decay_lr = False  # whether to decay the learning rate
args.epochs = 10000  # usually should be 5000-10000 epochs for convergent training
args.load_met = True
args.start_epoch = 0  # changed when loading pretrain met state file
args.last_epoch = 4999  # specify last epoch for loading met pretrain state file


# --------------------------------- output directory -------------------------------------#
args.out_dir = "gene_expr/"
args.out_filename = None  # generate out dirs automatically



