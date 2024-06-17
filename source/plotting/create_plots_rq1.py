import os
import glob
import scipy
import pickle
import numpy as np
import sys
import os
# Set the working directory

sys.path.append('c:\\Users\\david\\Dropbox\\GitHub\\OfflineRL')
sys.path.append('c:\\Users\\david\\Dropbox\\GitHub\\OfflineRL\\source')

# Print the current working directory to verify
print(sys.path)

from offline_ds_evaluation.metrics_manager import MetricsManager
from utils_gp import interquartile_mean
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
import pandas as pd
sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

offset_ann = 0.025
folder = "figures_rq_one"
image_type = "pdf"
"""
folder = "main_figures"
image_type = "png"
"""

figsize = (12, 6)
figsize_legend = (12, 1)
figsize_half = (12, 3.5)
figsize_half_half = (9.25, 3.5)
figsize_small_avg = (9, 3.2)
figsize_small = (13, 3.1)
figsize_comp = (12, 6)
figsize_comp_rot = (9, 8)
figsize_envs = (12, 7.2)
figsize_theplot = (13, 12)
figsize_thesmallplot = (9, 8)

# metric manager
experiments = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"]
experiments_gp = ["exgp_one_cart", "exgp_one_moun"]


mm = MetricsManager(0)
mm_GP = MetricsManager(0)

mm_dict = {"DQN":mm, "GP":mm_GP}

useruns = 5

#Get experiment data -> RL
for ex in experiments:
    for userun in range(1, 6):
        paths = glob.glob(os.path.join("data", ex, f"metrics_*_run{userun}.pkl"))
        for path in paths:
            with open(path, "rb") as f:
                m = pickle.load(f)
                m.recode(userun)
                print(path)
            mm.data.update(m.data)

#Get experiment data -> GP
for ex in experiments_gp:
    for userun in range(1, 6):
        paths = glob.glob(os.path.join("data", ex, f"metrics_*_run{userun}.pkl"))
        for path in paths:
            with open(path, "rb") as f:
                m = pickle.load(f)
                m.recode(userun)
                print(path)
            mm_GP.data.update(m.data)


# static stuff

envs = {
    'CartPole-v1': 0,
    'MountainCar-v0': 1
    }

algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]

buffer = {"random": "Random", "mixed": "Mixed", "er": "Replay", "noisy": "Noisy", "fully": "Expert"}

mc_actions = ["Acc. to the Left", "Don't accelerate", "Acc. to the Right"]

modes = list(buffer.keys())

markers = ["o", "D", "*", "<", "s"]

datasets = [Line2D([0], [0], color="black", marker=markers[i], linewidth=0) for i in range(len(markers))]


def plt_csv(ax, csv, algo, mode, ylims=None, set_title=True, color=None, set_label=True):
    est = np.mean(csv, axis=1)
    sd = np.std(csv, axis=1)
    cis = (est - sd, est + sd)

    ax.fill_between(np.arange(0, len(est) * 100, 100), cis[0], cis[1], alpha=0.2, color=color)
    ax.plot(np.arange(0, len(est) * 100, 100), est, label=(algo if set_label else None), color=color)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if set_title:
        ax.set_title(buffer[mode])
    if ylims != None:
        ax.set_ylim(bottom=ylims[0], top=ylims[1])


## get data
indir = os.path.join("..", "OfflineRL", "results", "csv_per_userun", "return")

files = []
for file in glob.iglob(os.path.join(indir, "**", "*csv"), recursive=True):
    files.append(file)
    print(file)

data = dict()
for file in files:
    name = file.split("\\")[-1]
    userun = int(file.split("\\")[-2][-1])
    env = file.split("\\")[-3]
    algo = name.split("_")[-2]
    mode = name.split("_")[-1].split(".")[0]


    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)
    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)
    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or userun not in data[env].keys():
        data[env][userun] = dict()
    if not data[env][userun].keys() or mode not in data[env][userun].keys():
        data[env][userun][mode] = dict()

    data[env][userun][mode][algo] = csv


# ###############
# # plot metrics for policies - boxplot over all datasets
# ###############

for coll_id, mm_var in mm_dict.items():
    outdir = os.path.join("..", "..", "results", folder, "Boxplots")
    os.makedirs(outdir, exist_ok=True)
    # titles
    x_label = "Dataset"

    # compact representation averaged over envs
    # f, axs = plt.subplots(1, 3, figsize=figsize_half, sharex=True)
    # for m, metric in enumerate([(0, 0), 2, (3, 0)]):
    #     x_all = []
    #     for mode in modes:
    #         x = []
    #         for e, env in enumerate(envs):
    #             for userun in range(1, useruns + 1):

    #                 random_return = mm.get_data(env, "random", userun)[0][0]
    #                 online_usap = mm.get_data(env, "er", userun)[2]

    #                 if m == 1:
    #                     result = mm_var.get_data(env, mode, userun)[metric]
    #                 else:
    #                     result = mm_var.get_data(env, mode, userun)[metric[0]][metric[1]]

    #                 if m == 0:
    #                     csv = data[env][userun]["online"]["DQN"]
    #                     x.append((result - random_return) / (np.max(csv) - random_return))
    #                 elif m == 1:
    #                     x.append(result / online_usap)
    #                 else:
    #                     x.append(result)

    #         x_all.append(x)

    #     axs[m].boxplot(x_all, positions=range(len(modes)), widths=0.4, zorder=20,
    #                 medianprops={"c": f"darkcyan", "linewidth": 1.2},
    #                 boxprops={"c": f"darkcyan", "linewidth": 1.2},
    #                 whiskerprops={"c": f"darkcyan", "linewidth": 1.2},
    #                 capprops={"c": f"darkcyan", "linewidth": 1.2},
    #                 flierprops={"markeredgecolor": f"darkcyan"})#, "markeredgewidth": 1.5})

    #     if m == 0:
    #         axs[m].set_ylabel("TQ", fontsize="medium")
    #         axs[m].axhline(y=1, color="silver")
    #     elif m == 1:
    #         axs[m].set_ylabel("SACo", fontsize="medium")
    #         axs[m].axhline(y=1, color="silver")
    #     elif m == 2:
    #         axs[m].set_ylabel("Entropy", fontsize="medium")

    #     axs[m].set_ylim(bottom=-0.05, top=1.45)
    #     axs[m].set_xticks([i for i in range(len(modes))])
    #     axs[m].set_xticklabels([buffer[m] for m in modes],fontsize="small")#, rotation=15, rotation_mode="anchor")
    # axs[-1].set_ylim(bottom=-0.05, top=1.05)
    # f.tight_layout(rect=(0, 0.022, 1, 1))
    # f.text(0.52, 0.01, x_label, ha='center')
    # plt.savefig(os.path.join(outdir, f"Boxplots_one_{coll_id}."))
    # plt.close()

    # compact representation averaged over envs -> with red dot for mean
    f, axs = plt.subplots(1, 2, figsize=figsize_small_avg, sharex=True)
    for m, metric in enumerate([(0, 0), 2]):
        x_all = []
        for mode in modes:
            x = []
            for e, env in enumerate(envs):
                for userun in range(1, useruns + 1):
                    random_return = mm.get_data(env, "random", 1)[0][0]
                    online_usap = mm.get_data(env, "er", 1)[2]
                    print("Online Usap: ", online_usap)

                    if m == 1:
                        result = mm_var.get_data(env, mode, userun)[metric]
                    else:
                        result = mm_var.get_data(env, mode, userun)[metric[0]][metric[1]]

                    if m == 0:
                        csv = data[env][userun]["online"]["DQN"]
                        x.append((result - random_return) / (np.max(csv) - random_return))
                    elif m == 1:
                        if mode == "er":
                            print(coll_id, env, userun, result, online_usap)
                        x.append(result / online_usap)

            x_all.append(x)

        axs[m].boxplot(x_all, positions=range(len(modes)), widths=0.4, zorder=20,
                    medianprops={"c": "darkcyan", "linewidth": 1.2},
                    boxprops={"c": "darkcyan", "linewidth": 1.2},
                    whiskerprops={"c": "darkcyan", "linewidth": 1.2},
                    capprops={"c": "darkcyan", "linewidth": 1.2},
                    flierprops={"markeredgecolor": "darkcyan"})#, "markeredgewidth": 1.5})
        axs[m].scatter(range(len(modes)), [np.mean(x_) for x_ in x_all], color="indianred")

        if m == 0:
            axs[m].set_ylabel("TQ", fontsize="medium")
            axs[m].axhline(y=1, color="silver")
        elif m == 1:
            axs[m].set_ylabel("SACo", fontsize="medium")
            axs[m].axhline(y=1, color="silver")

        axs[m].set_ylim(bottom=-0.05, top=1.45)
        axs[m].set_xticks([i for i in range(len(modes))])
        axs[m].set_xticklabels([buffer[m] for m in modes],fontsize="small")#, rotation=15, rotation_mode="anchor")

    f.tight_layout(rect=(0, 0.022, 1, 1))
    f.text(0.52, 0.01, x_label, ha='center', fontsize="medium")
    plt.savefig(os.path.join(outdir, f"Boxplots_two_{coll_id}"))
    plt.close()

# Mountain Car 5-Trajectories Plot                          ------------------------------------------------------------------------------------------------------------------------------------------------------
# Collection Algorithm: DQN
outdir = os.path.join("..", "..", "results", folder)
os.makedirs(outdir, exist_ok=True)

colors=["#39568CFF", "#ffcf20FF", "#29AF7FFF"]

samples = 10000

np.random.seed(42)
ind = np.random.choice(10**5, (samples, ), replace=False)

f, axs = plt.subplots(5, 5, figsize=figsize_theplot, sharex=True, sharey=True)
axs = [item for sublist in zip(axs[0], axs[1], axs[2], axs[3], axs[4]) for item in sublist]

for m, bt in enumerate(buffer):
    for userun in range(1, useruns + 1):
        ax = axs[m*5 + userun - 1]
        # load saved buffer
        with open(os.path.join("data", f"ex2", f"MountainCar-v0_run{userun}_{bt}.pkl"), "rb") as file:
            er_buffer = pickle.load(file)
            if userun == 1:
                ax.set_title(buffer[bt])
            ax.scatter(er_buffer.state[ind, 0], er_buffer.state[ind, 1], c=[colors[a] for a in er_buffer.action[ind, 0]], s=0.5)
            ax.text(0.02, 0.92, f"Run {userun}", fontsize="small", transform=ax.transAxes)

f.tight_layout(rect=(0.022, 0.022, 1, 0.97))
labels = [mpatches.Patch(color=colors[a], fill=True, linewidth=1, label=mc_actions[a]) for a in range(3)]
f.legend(handles=labels, handlelength=1, loc="upper right", ncol=3, fontsize="small")
f.text(0.53, 0.98, "Dataset", ha='center', fontsize="large")
f.text(0.53, 0.01, "Position in m", ha='center', fontsize="large")
f.text(0.005, 0.5, "Velocity in m/s", va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar.png"))
plt.close()

# Mountain Car 5-Trajectories Plot                          ------------------------------------------------------------------------------------------------------------------------------------------------------
# Collection Algorithm: GP
outdir = os.path.join("..", "..", "results", folder)
os.makedirs(outdir, exist_ok=True)

colors=["#39568CFF", "#ffcf20FF", "#29AF7FFF"]

samples = 10000

np.random.seed(42)
ind = np.random.choice(10**5, (samples, ), replace=False)

f, axs = plt.subplots(5, 5, figsize=figsize_theplot, sharex=True, sharey=True)
axs = [item for sublist in zip(axs[0], axs[1], axs[2], axs[3], axs[4]) for item in sublist]

for m, bt in enumerate(buffer):
    for userun in range(1, useruns + 1):
        ax = axs[m*5 + userun - 1]
        # load saved buffer
        with open(os.path.join("data", f"exgp_one_moun", f"MountainCar-v0_run{userun}_{bt}.pkl"), "rb") as file:
            er_buffer = pickle.load(file)
            if userun == 1:
                ax.set_title(buffer[bt])
            ax.scatter(er_buffer.state[ind, 0], er_buffer.state[ind, 1], c=[colors[a] for a in er_buffer.action[ind, 0]], s=0.5)
            ax.text(0.02, 0.92, f"Run {userun}", fontsize="small", transform=ax.transAxes)

f.tight_layout(rect=(0.022, 0.022, 1, 0.97))
labels = [mpatches.Patch(color=colors[a], fill=True, linewidth=1, label=mc_actions[a]) for a in range(3)]
f.legend(handles=labels, handlelength=1, loc="upper right", ncol=3, fontsize="small")
f.text(0.53, 0.98, "Dataset", ha='center', fontsize="large")
f.text(0.53, 0.01, "Position in m", ha='center', fontsize="large")
f.text(0.005, 0.5, "Velocity in m/s", va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar_GP.png"))
plt.close()


# Mountain Car Single Run Plot                          ------------------------------------------------------------------------------------------------------------------------------------------------------
# Collection Algorithm: DQN
f, axs = plt.subplots(1, 5, figsize=figsize_small, sharex=True, sharey=True)
userun = 1

for m, bt in enumerate(buffer):
    ax = axs[m]
    # load saved buffer
    with open(os.path.join("data", f"ex2", f"MountainCar-v0_run{userun}_{bt}.pkl"), "rb") as file:
        er_buffer = pickle.load(file)
        if userun == 1:
            ax.set_title(buffer[bt] + " Dataset")
        ax.scatter(er_buffer.state[ind, 0], er_buffer.state[ind, 1], c=[colors[a] for a in er_buffer.action[ind, 0]], s=0.5)

f.tight_layout(rect=(0.022, 0.08, 1, 1))
labels = [mpatches.Patch(color=colors[a], fill=True, linewidth=1, label=mc_actions[a]) for a in range(3)]
f.legend(handles=labels, handlelength=1, loc="lower right", ncol=3, fontsize="small")
#f.text(0.528, 0.9, "Dataset", ha='center', fontsize="large")
f.text(0.53, 0.04, "Position in m", ha='center', fontsize="large")
f.text(0.005, 0.5, "Velocity in m/s", va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar_small.png"))
plt.close()

# Mountain Car Single Run Plot                          ------------------------------------------------------------------------------------------------------------------------------------------------------
# Collection Algorithm: GP
f, axs = plt.subplots(1, 5, figsize=figsize_small, sharex=True, sharey=True)
userun = 1
for m, bt in enumerate(buffer):
    ax = axs[m]
    # load saved buffer
    with open(os.path.join("data", f"exgp_one_moun", f"MountainCar-v0_run{userun}_{bt}.pkl"), "rb") as file:
        er_buffer = pickle.load(file)
    if userun == 1:
        ax.set_title(buffer[bt] + " Dataset")
    ax.scatter(er_buffer.state[ind, 0], er_buffer.state[ind, 1], c=[colors[a] for a in er_buffer.action[ind, 0]], s=0.5)
f.tight_layout(rect=(0.022, 0.08, 1, 1))
labels = [mpatches.Patch(color=colors[a], fill=True, linewidth=1, label=mc_actions[a]) for a in range(3)]
f.legend(handles=labels, handlelength=1, loc="lower right", ncol=3, fontsize="small")
#f.text(0.528, 0.9, "Dataset", ha='center', fontsize="large")
f.text(0.53, 0.04, "Position in m", ha='center', fontsize="large")
f.text(0.005, 0.5, "Velocity in m/s", va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar_small_gp.png"))
plt.close()


# TQ vs SACo Plot                          ------------------------------------------------------------------------------------------------------------------------------------------------------
# Collection Algorithm: DP

max_return_table = {}
min_return_table = {}
average_return_table = {}
unique_sa_table = {}

for env, idx in envs.items():
    x_label = r"SACo of Dataset"
    y_label = r"TQ of Dataset"

    # Get metrics from reference data set from the DQN collection
    online_return_list=[]
    random_return_list=[]
    online_usap_list = []

    random_x, random_y, fully_x, fully_y, mixed_x, mixed_y, er_x, er_y, noisy_x, noisy_y = [], [], [], [], [], [], [], [], [], []
    random_x_gp, random_y_gp, fully_x_gp, fully_y_gp, mixed_x_gp, mixed_y_gp, er_x_gp, er_y_gp, noisy_x_gp, noisy_y_gp = [], [], [], [], [], [], [], [], [], []

    for userun in range(1,useruns+1):
        online_return_list.append(np.max(data[env][1]["online"]["DQN"]))
        random_return_list.append(mm.get_data(env, "random", 1)[0][0])
        online_usap_list.append(mm.get_data(env, "er", 1)[2])

        for m, mode in enumerate(["random", "fully", "mixed", "er", "noisy"]):
            if mode == "random":
                random_x.append(mm.get_data(env, mode, userun)[2])
                random_y.append(mm.get_data(env, mode, userun)[0][0])
                random_x_gp.append(mm_GP.get_data(env, mode, userun)[2])
                random_y_gp.append(mm_GP.get_data(env, mode, userun)[0][0])

            elif mode == "fully":
                fully_x.append(mm.get_data(env, mode, userun)[2])
                fully_y.append(mm.get_data(env, mode, userun)[0][0])
                fully_x_gp.append(mm_GP.get_data(env, mode, userun)[2])
                fully_y_gp.append(mm_GP.get_data(env, mode, userun)[0][0])

            elif mode == "mixed":
                mixed_x.append(mm.get_data(env, mode, userun)[2])
                mixed_y.append(mm.get_data(env, mode, userun)[0][0])
                mixed_x_gp.append(mm_GP.get_data(env, mode, userun)[2])
                mixed_y_gp.append(mm_GP.get_data(env, mode, userun)[0][0])
            elif mode == "er":
                er_x.append(mm.get_data(env, mode, userun)[2])
                er_y.append(mm.get_data(env, mode, userun)[0][0])
                er_x_gp.append(mm_GP.get_data(env, mode, userun)[2])
                er_y_gp.append(mm_GP.get_data(env, mode, userun)[0][0])
            elif mode == "noisy":
                noisy_x.append(mm.get_data(env, mode, userun)[2])
                noisy_y.append(mm.get_data(env, mode, userun)[0][0])
                noisy_x_gp.append(mm_GP.get_data(env, mode, userun)[2])
                noisy_y_gp.append(mm_GP.get_data(env, mode, userun)[0][0])


    online_return_iqm = interquartile_mean(online_return_list)
    random_return_iqm = interquartile_mean(random_return_list)
    online_usap_iqm = interquartile_mean(online_usap_list)
    random_x_iqm = interquartile_mean(random_x)
    random_y_iqm = interquartile_mean(random_y)
    fully_x_iqm = interquartile_mean(fully_x)
    fully_y_iqm = interquartile_mean(fully_y)
    mixed_x_iqm = interquartile_mean(mixed_x)
    mixed_y_iqm = interquartile_mean(mixed_y)
    er_x_iqm = interquartile_mean(er_x)
    er_y_iqm = interquartile_mean(er_y)
    noisy_x_iqm = interquartile_mean(noisy_x)
    noisy_y_iqm = interquartile_mean(noisy_y)
    random_x_gp_iqm = interquartile_mean(random_x_gp)
    random_y_gp_iqm = interquartile_mean(random_y_gp)
    fully_x_gp_iqm = interquartile_mean(fully_x_gp)
    fully_y_gp_iqm = interquartile_mean(fully_y_gp)
    mixed_x_gp_iqm = interquartile_mean(mixed_x_gp)
    mixed_y_gp_iqm = interquartile_mean(mixed_y_gp)
    er_x_gp_iqm = interquartile_mean(er_x_gp)
    er_y_gp_iqm = interquartile_mean(er_y_gp)
    noisy_x_gp_iqm = interquartile_mean(noisy_x_gp)
    noisy_y_gp_iqm = interquartile_mean(noisy_y_gp)

    x, y = [], []
    for m, mode in enumerate(["random", "fully", "mixed", "er", "noisy"]):
        if mode == "random":
            x.append(random_x_iqm / online_usap_iqm) #SACO
            y.append((random_y_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "fully":
            x.append(fully_x_iqm / online_usap_iqm) #SACO
            y.append((fully_y_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "mixed":
            x.append(mixed_x_iqm / online_usap_iqm) #SACO
            y.append((mixed_y_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "er":
            x.append(er_x_iqm / online_usap_iqm) #SACO
            y.append((er_y_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "noisy":
            x.append(noisy_x_iqm / online_usap_iqm) #SACO
            y.append((noisy_y_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ


    f = plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s = 70)
    plt.xlim(-0.02,1.25)
    plt.ylim(-0.02,1.25)

    for i, annotation in enumerate(['Random', 'Expert', 'Mixed', 'Replay ', 'Noisy']):
        plt.annotate(annotation,
                    (x[i], y[i] + offset_ann - 0.002),
                    fontsize="medium", va="bottom", ha="center",zorder=20)

    f.tight_layout(rect=(0.022, 0.022, 1, 1))
    f.text(0.5, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"{env}_tqvssaco" + image_type))
    plt.close()

    # TQ vs SACo Plot                          ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Collection Algorithm: GP

    x_label = r"SACo of Dataset"
    y_label = r"TQ of Dataset"

    #How is the mm constructed:
    #returns,       unique_states, unique_state_actions, entropies, sparsities, ep_lengths
    #0-> mean, 1 std

    x_gp, y_gp = [], []
    for m, mode in enumerate(["random", "fully", "mixed", "er", "noisy"]):
        if mode == "random":
            x_gp.append(random_x_gp_iqm / online_usap_iqm) #SACO
            y_gp.append((random_y_gp_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "fully":
            x_gp.append(fully_x_gp_iqm / online_usap_iqm) #SACO
            y_gp.append((fully_y_gp_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "mixed":
            x_gp.append(mixed_x_gp_iqm / online_usap_iqm) #SACO
            y_gp.append((mixed_y_gp_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "er":
            x_gp.append(er_x_gp_iqm / online_usap_iqm) #SACO
            y_gp.append((er_y_gp_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ
        elif mode == "noisy":
            x_gp.append(noisy_x_gp_iqm / online_usap_iqm) #SACO
            y_gp.append((noisy_y_gp_iqm - random_return_iqm) / (online_return_iqm - random_return_iqm)) #TQ

    f = plt.figure(figsize=(6, 5))
    plt.scatter(x_gp, y_gp, s = 70)
    plt.xlim(-0.02, 1.25)
    plt.ylim(-0.02, 1.25)

    for i, annotation in enumerate(['Random', 'Expert', 'Mixed', 'Replay  ', 'Noisy']):
        plt.annotate(annotation,
                    (x_gp[i], y_gp[i] + offset_ann - 0.002),
                    fontsize="medium", va="bottom", ha="center",zorder=20)


    f.tight_layout(rect=(0.022, 0.022, 1, 1))
    f.text(0.5, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    #plt.savefig(os.path.join(outdir, f"{env}_tqvssaco_GP" + image_type))
    plt.close()

    # Create a dictionary with the lists
    list_dict = {
        'online_return': online_return_list,
        'random_return': random_return_list,
        'online_usap': online_usap_list,
        'random_x': random_x,
        'random_y': random_y,
        'fully_x': fully_x,
        'fully_y': fully_y,
        'mixed_x': mixed_x,
        'mixed_y': mixed_y,
        'er_x': er_x,
        'er_y': er_y,
        'noisy_x': noisy_x,
        'noisy_y': noisy_y,
        'random_x_gp': random_x_gp,
        'random_y_gp': random_y_gp,
        'fully_x_gp': fully_x_gp,
        'fully_y_gp': fully_y_gp,
        'mixed_x_gp': mixed_x_gp,
        'mixed_y_gp': mixed_y_gp,
        'er_x_gp': er_x_gp,
        'er_y_gp': er_y_gp,
        'noisy_x_gp': noisy_x_gp,
        'noisy_y_gp': noisy_y_gp,
        'TQ_DQN': x,
        'SACo_DQN': y,
        'TQ_GP': x_gp,
        'SACo_GP': y_gp
    }


    # Save the metris for the Appendix:
    #   Fixed
    max_return_table[env] = online_return_list
    min_return_table[env] = random_return_list
    #   DQN
    average_return_table[env] = {"random": random_y, "fully": fully_y, "mixed": mixed_y, "er": er_y, "noisy": noisy_y}
    unique_sa_table[env] = {"random": random_x, "fully": fully_x, "mixed": mixed_x, "er": er_x, "noisy": noisy_x}
    #  GP
    average_return_table[env+"_GP"] = {"random": random_y_gp, "fully": fully_y_gp, "mixed": mixed_y_gp, "er": er_y_gp, "noisy": noisy_y_gp}
    unique_sa_table[env+"_GP"] = {"random": random_x_gp, "fully": fully_x_gp, "mixed": mixed_x_gp, "er": er_x_gp, "noisy": noisy_x_gp}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list_dict)

    # Save DataFrame as Excel
    excel_path = os.path.join(outdir, f"{env}_tqvssaco.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"DataFrame saved as Excel at {excel_path}")

df_max_return = pd.DataFrame(max_return_table)
df_min_return = pd.DataFrame(min_return_table)
df_average_return = pd.DataFrame(average_return_table)
df_unique_sa = pd.DataFrame(unique_sa_table)

# Create the directory if it does not exist
outdir = os.path.join("..", "..", "results", folder, "appendix")
os.makedirs(outdir, exist_ok=True)
excel_path = os.path.join(outdir, "max_return.xlsx")
df_max_return.to_excel(excel_path, index=False)

excel_path = os.path.join(outdir, "min_return.xlsx")
df_min_return.to_excel(excel_path, index=False)

excel_path = os.path.join(outdir, "average_return.xlsx")
df_average_return.to_excel(excel_path, index=False)

excel_path = os.path.join(outdir, "unique_sa.xlsx")
df_unique_sa.to_excel(excel_path, index=False)
print("Excel files saved in the appendix folder")
