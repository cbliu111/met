import os
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import torch
from met.met import MET

from models.autoreg import Model, args

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_LAUNCH_BLOCKING=1



path = "AutoReg_final.pt"
state = torch.load(path)
net = MET(**vars(args))
net.load_state_dict(state["net_state_dict"])
net.to(args.device)
net.eval()

bs = 1000

su = 1
ru = 1
sb = np.linspace(1e-8, 0.1, 100)
rb = np.linspace(1e-8, 100, 100)
bc_list = []
for i in sb:
    for j in rb:
        rate = np.array([su, i, ru, j, 1])
        t = 10
        with torch.no_grad():
            init = torch.as_tensor(args.init, dtype=args.default_dtype_torch, device=args.device)
            rate = torch.as_tensor(rate, dtype=args.default_dtype_torch, device=args.device)
            t += args.epsilon
            t = torch.as_tensor(t, dtype=args.default_dtype_torch, device=args.device).view(1)
            prompt = torch.cat((torch.log(rate.view(-1)), init.view(-1), t.view(-1)), dim=0)
            print(f"rate: {rate}")
            prompt = prompt.repeat(bs, 1)
            samples = []
            for j in range(1):
                _, sample = net.sample(prompt)
                samples.append(sample.detach().cpu().numpy())
            x = np.concatenate(samples, axis=0)
            x1 = x[:, 1]  # protein
            m3 = skew(x1, bias=False)
            m4 = kurtosis(x1, fisher=True, bias=False)
            n = x1.shape[0]

            # bc = (m3 ** 2 + 1) / (m4 + 3 * (n - 1) ** 2 / (n - 2) / (n - 3))
            bc = 1 / (m4 + 3 - m3 ** 2)
            print(f"bc is: {bc}")
            bc_list.append(bc)

np.savez("autoreg_met_bc_sample", bc=np.array(bc_list), sb=sb, rb=rb)
