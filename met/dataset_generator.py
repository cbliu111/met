import time
import os
import numpy as np
import torch
from torch import nn
import copy
from met.cme_solver import cme_state_joint_prob

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def ensure_dir(filename):
    """
    make sure directory exists
    """
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

def construct_optimizer(net, args):
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    optimizer = torch.optim.Adam(params, lr=args.van_lr, betas=(0.9, 0.999))

    return optimizer, params, nparams


def generate(model, net, case: int, init: list, rates, args):

    ensure_dir(args.out_dir + "dataset/")
    ensure_dir(args.out_dir + "dataset/checkpoints/")
    ensure_dir(args.out_dir + "dataset/loss/")
    ensure_dir(args.out_dir + "dataset/results/")

    start_time = time.time()
    default_dtype_torch = args.default_dtype_torch

    if args.model != 'Epidemic':
        reaction_mat_left, reaction_mat_right = model.reaction_matrix()
        rate_parameters = torch.as_tensor(rates, dtype=default_dtype_torch).to(args.device)
        transition_matrix = reaction_mat_right - reaction_mat_left  # transition matrix, species x reactions
    else:
        rate_parameters = None
        reaction_mat_left = None
        transition_matrix = None

    # assign init to args
    args.initial_distribution = np.array(init).reshape(1, -1)
    args.initial_poisson_param = np.array(init)

    args.max_step = args.max_step_initial_training

    # optimizer
    optimizer, params, nparams = construct_optimizer(net, args)

    # iterate over all cme steps
    cme_start_step = 0
    cme_time = 0
    delta_t = args.cme_dt
    new_net = []
    all_loss_mean = []  # cme_step x train_step, batch-mean loss
    all_loss_std = []  # cme_step x train_step, batch-std loss
    all_kl = []
    all_ed = []
    all_sample = []  # cme_step%print_step x 10*batch_size x num_species
    cme_times = []  # delta_t from cme_step_solver
    for cme_step in range(cme_start_step, args.total_cme_steps):  # Time train_step of the dynamical equation
        step_start_time = time.time()
        if args.model == 'Epidemic':
            reaction_mat_left, reaction_mat_right = model.reaction_matrix()
            rate_parameters = torch.as_tensor(rates, dtype=default_dtype_torch, device=args.device)
            transition_matrix = reaction_mat_right - reaction_mat_left  # transition matrix, species x reactions

        scheduler = None
        if args.lr_schedule:
            if args.lr_schedule_type == 1:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.5,
                    patience=int(args.max_step * args.epoch_percent),
                    verbose=True,
                    threshold=1e-4,
                    min_lr=1e-5
                )
            if args.lr_schedule_type == 2:
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: 1 / (epoch * 10 * args.lr + 1)
                )

        if cme_step >= 1:
            args.max_step = args.max_step_training

        step_lm = []
        step_ls = []
        step_kl = []
        step_ed = []
        step_sample = []
        for train_step in range(args.max_step + 1):
            optimizer.zero_grad()
            with torch.no_grad():
                sample, x_hat = net.sample(args.van_batch_size)

            log_prob = net.log_prob(sample)
            if train_step == 0:
                delta_t = 1
            with torch.no_grad():
                log_Tp_t, delta_t = cme_state_joint_prob(
                    sample,
                    args,
                    cme_step,
                    train_step,
                    new_net,
                    reaction_mat_left,
                    transition_matrix,
                    rate_parameters,
                    delta_t,
                    model.propensity_function,
                )  # .detach()
                self_norm = torch.exp(log_Tp_t) / (torch.exp(log_Tp_t)).sum()
                # account for normalization difference
                Tp_t_normalize = (self_norm * (torch.exp(log_prob)).sum()).detach()
                # prob ratio
                loss = log_prob - log_Tp_t.detach()
                # prob difference
                loss_l2 = (torch.exp(log_prob) - Tp_t_normalize)
                # square root of prob product
                loss_he = -torch.sqrt(torch.exp(log_prob) * Tp_t_normalize)

            assert not log_Tp_t.requires_grad

            if args.loss_type == 'kl':
                loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            elif args.loss_type == 'klreweight':
                loss3 = torch.exp(log_prob) * loss / torch.exp(log_prob).mean()
                loss_reinforce = torch.mean((loss3 - loss3.mean()) * log_prob)
            elif args.loss_type == 'l2':
                loss_reinforce = torch.mean((loss_l2 - loss_l2.mean()) * log_prob)
            elif args.loss_type == 'he':
                loss_reinforce = torch.mean(loss_he * log_prob)
            elif args.loss_type == 'ss':
                # steady state# Conversion from probability P to state \psi
                loss_reinforce = torch.mean((loss - loss.mean()) * log_prob / 2)
            else:
                raise ValueError('Unknown loss type: {}'.format(args.loss_type))

            loss_reinforce.backward()

            if args.clip_grad:
                nn.utils.clip_grad_norm_(params, args.clip_grad)
            optimizer.step()
            if args.lr_schedule:
                scheduler.step(loss.mean())

            loss_std = loss.std()
            loss_mean = loss.mean()
            step_lm.append(loss_mean.detach().cpu().numpy())
            step_ls.append(loss_std.detach().cpu().numpy())
            f = torch.exp(net.log_prob(sample)) - Tp_t_normalize
            f = f ** 2
            euclidean_distance = torch.sqrt(torch.sum(f))
            kl_distance = torch.nn.functional.kl_div(
                net.log_prob(sample),
                Tp_t_normalize,
                None,
                None,
                'sum'
            )  # function kl_div is not the same as wiki's explanation.

            step_ed.append(euclidean_distance.detach().cpu().numpy())
            step_kl.append(kl_distance.detach().cpu().numpy())

        # save net state and samples at print step
        with torch.no_grad():
            for _ in range(10):
                with torch.no_grad():
                    step_sample.append(sample.detach().cpu().numpy())
            # record loss and distance for every train_step
            all_loss_mean.append(np.mean(step_lm))
            all_loss_std.append(np.mean(step_ls))
            all_ed.append(np.mean(step_ed))
            all_kl.append(np.mean(step_kl))
            # record delta_t at all steps
            cme_time += delta_t
            # copy net for next cme step
            new_net = copy.deepcopy(net)  # net
            new_net.requires_grad = False
            if cme_step % args.save_step == 0:
                all_sample.append(np.concatenate(step_sample, axis=0))
                cme_times.append(cme_time)
                path = f"{args.out_dir}/dataset/checkpoints/{args.model}_case{case}_step{cme_step}.pt"
                torch.save({
                    "model": args.model,
                    "net": args.net,
                    "rate_parameters": rate_parameters,  # should save the current rates
                    "init": init,
                    "cme_time": cme_time,
                    "net_state_dict": net.state_dict(),
                    "dtype": args.dtype,
                    "optimizer": optimizer.state_dict(),
                    "case": case,
                    "trained_steps": args.max_step,
                }, path)
                path1 = f"{args.out_dir}/dataset/loss/{args.model}_case{case}_step{cme_step}"
                np.savez(
                    path1,
                    loss_mean=np.array(step_lm),
                    loss_std=np.array(step_ls),
                    ed=np.array(step_ed),
                    kl=np.array(step_kl)
                )

        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        if cme_step % args.print_step == 0:
            ss = np.array(step_sample).reshape(-1, args.num_species)
            print(f"current cme_time is: {cme_time}, species mean {ss.mean(axis=0)}, species s.d. {ss.std(axis=0)}")
            rt = step_time * (args.end_time/delta_t - cme_step) / 3600
            print(f"one cme step used {step_time/60} min, estimated rest time is {rt} hr")

        # end iteration when passing end time
        if cme_time > args.end_time:
            break

    print('all_samples.shape', np.array(all_sample).shape)
    stop_time = time.time()
    total_run_time = np.array((stop_time - start_time) / 3600)
    print('Time(min) ', total_run_time * 60)
    print('Time(hr) ', total_run_time)

    np.savez(
        f'{args.out_dir}/dataset/results/all_samples_case{case}',
        rates=rate_parameters.detach().cpu().numpy(),
        init=np.array(init),
        run_time=total_run_time,
        samples=np.array(all_sample),
        times=np.array(cme_times),
        loss_mean=np.array(all_loss_mean),
        loss_std=np.array(all_loss_std),
        ed=np.array(all_ed),
        kl=np.array(all_kl),
    )


