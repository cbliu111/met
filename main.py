import os
from met.dataset_generator import generate
import torch

from met.met import MET
from met.gru import GRU
from met.transformer import TraDE
from met.trainer import train

MODEL = "ToggleSwitch"

if MODEL == "AutoReg":
    from models.autoreg import Model, args
elif MODEL == "BirthDeath":
    from models.birth_death import Model, args
elif MODEL == "GeneExpr":
    from models.gene_expr import Model, args
elif MODEL == "LotkaVolterra":
    from models.lotka_volterra import Model, args
elif MODEL == "ToggleSwitch":
    from models.toggle_switch import Model, args
elif MODEL == "RNATurnover":
    from models.rna_turnover import Model, args
elif MODEL == "Cascade":
    from models.cascade import Model, args
elif MODEL == "Transcription":
    from models.transcription import Model, args
else:
    raise KeyError("Model not included.")

cme_model = Model(**vars(args))

if args.net == "rnn":
    reward_net = GRU(**vars(args))
elif args.net == "transformer":
    reward_net = TraDE(**vars(args))
else:
    raise KeyError("Unknown reward net type.")

reward_net.to(args.device)

rates = cme_model.rates()
init = args.init
args.batch_size = args.van_batch_size
generate(cme_model, reward_net, 0, init, rates, args)

# Initialize net, optimizer and model
met_net = MET(**vars(args))
met_net.to(args.device)
optim = met_net.configure_optimizers(
    args.weight_decay,
    args.lr,
    (args.beta1, args.beta2),
    device_type=args.device
)

if args.load_met:
    states = []
    epochs = []
    last_state = None
    for state in os.listdir(f"{args.out_dir}/met/train_state/"):
        epoch = state.split("_")[-1].replace("epoch", "").replace(".pt", "")
        if epoch == "final":
            last_state = f"{args.out_dir}/met/train_state/" + state
            break
        states.append(f"{args.out_dir}/met/train_state/" + state)
        epochs.append(int(epoch))
    if not last_state:
        last_epoch = max(epochs)
        args.start_epoch = last_epoch
        print(f"pretrain met state file found, last epoch: {last_epoch}")
        last_state = states[epochs.index(last_epoch)]
    else:
        last_epoch = args.last_epoch
        args.start_epoch = last_epoch
        print(f"pretrain met state file found, last epoch: {last_epoch}")

    s = torch.load(last_state)
    met_net.load_state_dict(s["net_state_dict"])
    optim.load_state_dict(s["optimizer_state_dict"])
    print(f"using pretrain met state for further training.")

directory = f"{args.out_dir}/dataset/checkpoints/"
data = []
for n_file, checkpoint in enumerate(os.listdir(directory)):
    checkpoint = directory + checkpoint
    data.append(checkpoint)

# start iterating
args.batch_size = args.met_batch_size
args.use_adaptive_cme_time = False
train(cme_model, data, met_net, reward_net, optim, args)

