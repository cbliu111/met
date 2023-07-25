import os
import numpy as np
import torch
from met.met import MET

from models.autoreg import Model, args

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_LAUNCH_BLOCKING=1

file = "autoreg_infer_case0.npz"
samples = np.load(file)["data"]  # batch x time x species
print(samples.shape)

path = "AutoReg_final.pt"
state = torch.load(path)
net = MET(**vars(args))
net.load_state_dict(state["net_state_dict"])
net.to(args.device)
net.eval()

cme_times = np.linspace(0, 10, 10)

bs = 1000

rub = torch.tensor([2, 0.1, 10, 100], device=args.device)
rlb = torch.tensor([1e-8, 1e-8, 1e-8, 1e-8], device=args.device)
r0 = [1, 0.05, 5, 50]
epochs = 10000
init = torch.as_tensor(args.init, dtype=args.default_dtype_torch, device=args.device)
rate = torch.tensor(r0, dtype=args.default_dtype_torch, device=args.device)
lr = np.array([1, 0.01, 1, 5])
rate_samples = []
with torch.no_grad():
    for i in range(epochs):
        sidx = np.random.choice(10000, bs, replace=False)
        tidx = np.random.choice(10, 1)
        x = samples[sidx, tidx, :]
        x = torch.as_tensor(x, dtype=args.default_dtype_torch, device=args.device)
        dr = lr * np.random.standard_normal(4)
        dr = torch.as_tensor(dr, device=args.device)
        new_rate = rate + dr
        out_idx = new_rate > rub
        new_rate[out_idx] = rub[out_idx]
        out_idx = new_rate < rlb
        new_rate[out_idx] = rlb[out_idx]
        r = torch.cat((rate, torch.ones(1, device=args.device)), dim=0)
        nr = torch.cat((new_rate, torch.ones(1, device=args.device)), dim=0)
        t = cme_times[tidx] + args.epsilon
        t = torch.as_tensor(t, dtype=args.default_dtype_torch, device=args.device).view(1)
        prompt = torch.cat((torch.log(r.view(-1)), init.view(-1), t.view(-1)), dim=0)
        nprompt = torch.cat((torch.log(nr.view(-1)), init.view(-1), t.view(-1)), dim=0)
        prompt = prompt.repeat(bs, 1)
        nprompt = nprompt.repeat(bs, 1)
        sp = torch.cat((prompt, x), dim=1)
        nsp = torch.cat((nprompt, x), dim=1)
        log_prob = net.log_joint_prob(sp)
        prob = torch.exp(log_prob)
        nlog_prob = net.log_joint_prob(nsp)
        nprob = torch.exp(nlog_prob)
        gamma = nprob / prob
        a = gamma[gamma>1].shape[0]
        b = gamma[gamma<1].shape[0]
        pth = a / b if b > 0 else 1
        pselect = np.random.rand()
        if pselect < pth:
            rate = new_rate

        print(f"epoch: {i}, rate: {rate}")
        rate_samples.append(rate.detach().cpu().numpy())


np.savez("autoreg_inference_rate_sample", data=np.array(rate_samples))
