import os
import math
import time
import torch
import random
import numpy as np
from torch import nn
from met.cme_solver import cme_state_joint_prob

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_lr(it):
    # adamw optimizer
    learning_rate = 1e-3  # max learning rate
    warmup_iters = 20  # how many steps to warm up for
    lr_decay_iters = 1000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizer(net, learning_rate):
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    num_params = int(sum([np.prod(p.shape) for p in params]))
    # adam = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    adam = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999))

    return adam, params, num_params


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, dataset, net, std_net, optimizer, args):

    ensure_dir(args.out_dir + "met/")
    ensure_dir(args.out_dir + "met/train_state/")
    ensure_dir(args.out_dir + "met/train_loss/")

    start_time = time.time()

    loss_means = []
    loss_stds = []
    eds = []
    kls = []

    reaction_mat_left, reaction_mat_right = model.reaction_matrix()
    transition_matrix = reaction_mat_right - reaction_mat_left  # transition matrix, species x reactions

    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        # set warmup learning rate
        # lr = get_warmup_learning_rate(epoch)
        lr = get_lr(epoch) if args.decay_lr else args.lr
        set_learning_rate(optimizer, lr)

        # prepare for new epoch
        random.shuffle(dataset)

        loss_mean_collection = []  # cme_step x train_step, batch-mean loss
        loss_std_collection = []  # cme_step x train_step, batch-std loss
        kl_distance_collection = []
        euclidean_distance_collection = []

        num_prompts = 100
        accumulation_steps = 1
        optimizer.zero_grad()  # accumulate weights for several prompts with small batch
        for n_file, checkpoint in enumerate(dataset[:num_prompts]):
            # load state to pretrained net
            std_state = torch.load(checkpoint)
            std_net.load_state_dict(std_state["net_state_dict"])
            std_net.eval()

            # construct prompt
            with torch.no_grad():
                init = torch.as_tensor(std_state["init"], dtype=args.default_dtype_torch, device=args.device)
                rates = std_state["rate_parameters"]
                rates = torch.as_tensor(rates, dtype=args.default_dtype_torch, device=args.device)
                t = std_state["cme_time"] + args.cme_dt
                t = torch.as_tensor(t, dtype=args.default_dtype_torch, device=args.device)
                prompt = torch.cat((torch.log(rates.view(-1)), init.view(-1), t.view(-1)), dim=0)
                # print(f"prompt: {prompt}")
                prompt = prompt.repeat(args.batch_size, 1)

            # accumulate weight for small batch
            for _ in range(accumulation_steps):
                with torch.no_grad():
                    pro_sam, samples = net.sample(prompt)  # pro_sam: prompt + samples as one complete sentence

                log_prob = net.log_joint_prob(pro_sam)

                with torch.no_grad():
                    log_Tp_t, _ = cme_state_joint_prob(
                        samples,
                        args,
                        epoch,
                        n_file,
                        std_net,
                        reaction_mat_left,
                        transition_matrix,
                        rates,
                        args.cme_dt,
                        model.propensity_function
                    )
                    prob = torch.exp(log_prob.detach())
                    r_prob = torch.exp(log_Tp_t.detach())
                    r_prob = r_prob / r_prob.sum()
                    r_prob = (r_prob * prob.sum()).detach()
                    loss = log_prob - log_Tp_t.detach()
                    loss_l2 = prob - r_prob
                    loss_he = -torch.sqrt(prob * r_prob)

                assert not log_Tp_t.requires_grad

                if args.loss_type == 'kl':
                    loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
                elif args.loss_type == 'klreweight':
                    loss3 = prob * loss / prob.mean()
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

                loss_std = loss.std()
                loss_mean = loss.mean()
                loss_mean_collection.append(loss_mean.detach().cpu().numpy())
                loss_std_collection.append(loss_std.detach().cpu().numpy())
                f = (prob - r_prob) ** 2
                euclidean_distance = torch.sqrt(torch.sum(f))
                kl_distance = torch.nn.functional.kl_div(
                    log_prob,
                    r_prob,
                    None,
                    None,
                    'sum'
                )
                euclidean_distance_collection.append(euclidean_distance.detach().cpu().numpy())
                kl_distance_collection.append(kl_distance.detach().cpu().numpy())

        if args.clip_grad:
            nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)

        optimizer.step()

        lm = np.mean(loss_mean_collection)
        ls = np.mean(loss_std_collection)
        ed = np.mean(euclidean_distance_collection)
        kl = np.mean(kl_distance_collection)
        loss_means.append(lm)
        loss_stds.append(ls)
        eds.append(ed)
        kls.append(kl)
        print(
            f"|epoch {epoch}|mean loss: {lm:.4f}, std loss: {ls:.4f}, ed: {ed:.4f}, kd: {kl:.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}")

        epoch_end_time = time.time()
        used_time = epoch_end_time - epoch_start_time
        finished_time = used_time * (args.epochs - epoch)

        print(f"One epoch use time {used_time / 60 : .2f} min, estimated rest time {finished_time / 3600 : .2f} hr")

        if epoch % 100 == 0:
            with torch.no_grad():
                path = f"{args.out_dir}/met/train_state/{args.model}_epoch{epoch}.pt"
                torch.save({
                    "net_state_dict": net.state_dict(),
                    "dtype": args.dtype,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, path)
                path1 = f"{args.out_dir}/met/train_loss/{args.model}_epoch{epoch}.npz"
                np.savez(
                    path1,
                    loss_mean=np.array(loss_means),
                    loss_std=np.array(loss_stds),
                    ed=np.array(eds),
                    kl=np.array(kls),
                )

    stop_time = time.time()
    print('Time(min) ', (stop_time - start_time) / 60)
    print('Time(hr) ', (stop_time - start_time) / 3600)
    path = f"{args.out_dir}/met/train_state/{args.model}_final.pt"
    torch.save({
        "net_state_dict": net.state_dict(),
        "dtype": args.dtype,
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


