import os
import numpy as np
import torch
from met.met import MET

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

met_times = []
met_samples = []

path = "final.pt"
state = torch.load(path)
net = MET(**vars(args))
net.load_state_dict(state["net_state_dict"])
net.to(args.device)
net.eval()

bs = 1000

for checktime in np.linspace(0, 10, 25):
    # gene_expr: 3600
    # toggle: 40
    print(f"handling {checktime}")
    with torch.no_grad():
        init = torch.as_tensor(args.init, dtype=args.default_dtype_torch, device=args.device)
        rates = args.rate_parameters
        rates = torch.as_tensor(rates, dtype=args.default_dtype_torch, device=args.device)
        t = checktime
        met_times.append(t)
        t += args.epsilon
        t = torch.as_tensor(t, dtype=args.default_dtype_torch, device=args.device).view(1)
        prompt = torch.cat((torch.log(rates.view(-1)), init.view(-1), t.view(-1)), dim=0)
        # print(f"prompt: {prompt}")
        prompt = prompt.repeat(bs, 1)

        samples = []
        for i in range(10):  # sampling 10 x batch_size samples for each time point
            _, sample = net.sample(prompt)
            samples.append(sample.detach().cpu().numpy())

        met_samples.append(np.concatenate(samples, axis=0))

np.savez("met_samples", samples=np.array(met_samples), times=np.array(met_times))

