from met.met import MET
from models.birth_death import args
import torch
import numpy as np

met_net = MET(**vars(args))
state = torch.load("Birth-Death_final.pt")
met_net.load_state_dict(state["net_state_dict"])
met_net.to(args.device)
met_net.eval()

time_step = 5

end_time = 100
bs = 10
times = []
samples = []
for _ in range(100):  # num trajs
    init = np.zeros((bs, 1), dtype=int)
    step_samples = []
    for i in range(int(end_time/time_step)):
        if i > 0:
            init = sample

        init = torch.as_tensor(init, dtype=args.default_dtype_torch, device=args.device)
        rates = args.rate_parameters
        rates = torch.as_tensor(rates, dtype=args.default_dtype_torch, device=args.device)
        rates = rates.repeat(bs, 1)
        t = time_step + args.epsilon
        t = torch.as_tensor(t, dtype=args.default_dtype_torch, device=args.device)
        t = t.repeat(bs, 1)
        prompt = torch.cat((torch.log(rates), init, t), dim=1)

        sample = met_net.sample(prompt)  # pro_sam: prompt + samples as one complete sentence

        step_samples.append(sample.detach().cpu().numpy().reshape(-1, 1))

    samples.append(np.concatenate(step_samples, axis=1))


times = np.array([time_step + i * time_step for i in range(int(end_time/time_step))])
samples = np.concatenate(samples, axis=0)  # trajs x times x species

np.savez("../autodl-tmp/birth_death/test2/traj_samples1", samples=samples, times=times)

