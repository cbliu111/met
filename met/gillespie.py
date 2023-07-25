import numpy as np
import biocircuits


def sim(model, args):
    # parameters for simulation
    reaction_mat_left, reaction_mat_right = model.reaction_matrix()
    sim_trans_matrix = (reaction_mat_right.detach().cpu().numpy() - reaction_mat_left.detach().cpu().numpy()).T  # reaction x species

    end_time = args.end_time
    trajectories = args.num_trajectories
    cores = args.cpu_cores
    slices = args.record_time_slices
    run = args.run_sim
    save = args.save_sim
    rate_parameters = args.rate_parameters

    # initial number of species
    pop0 = np.array(args.init, dtype=int)

    # simulation time length
    T = end_time
    time_points = np.linspace(0, T, slices)
    runs = int(trajectories / cores)
    if not args.save_sim_file:
        out_filename = f"sim_trajs{runs * cores}_end{end_time}"  # filename to save
    else:
        out_filename = args.save_sim_file

    samples = np.empty(shape=(trajectories, len(time_points), args.num_species), dtype=int)

    if run:
        samples = biocircuits.gillespie_ssa(
            model.sim_propensity_function,
            sim_trans_matrix,
            pop0,
            time_points,
            size=runs,
            args=rate_parameters,
            n_threads=cores,
            progress_bar=True,
        )
        if save:
            np.savez(out_filename, data=samples)  # sava data
    else:  # load existing data file
        a = np.load(out_filename)
        if "data" in a:
            samples = a["data"]
        elif "arr_0" in a:
            samples = a["arr_0"]
        print(f"shape of the samples: {samples.shape}")

    return samples


