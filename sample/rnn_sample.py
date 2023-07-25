import os
import numpy as np
import torch
from met.gru import GRU

MODEL = "RNATurnover"

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
else:
    raise KeyError("Model not included.")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_LAUNCH_BLOCKING=1

rnn_times = []
rnn_samples = []

bs = 10000

net = GRU(**vars(args))
path = "dataset/checkpoints/"
files = [f"case0_step{i}.pt" for i in range(0, 1000, 40)]
for file in files:
    print(f"handling {file}")
    state = torch.load(path + file)
    net.load_state_dict(state["net_state_dict"])
    net.to(args.device)
    net.eval()
    rnn_times.append(state["cme_time"])
    sample, _ = net.sample(bs)
    rnn_samples.append(sample.detach().cpu().numpy())  # times x batch_size x species

np.savez("rnn_samples", samples=np.array(rnn_samples), times=np.array(rnn_times))

