import pandas as pd
from  matplotlib.colors import LinearSegmentedColormap
import json
import os
import sys
# Set the working directory
sys.path.append('c:\\Users\\david\\Dropbox\\GitHub\\OfflineRL')
sys.path.append('c:\\Users\\david\\Dropbox\\GitHub\\OfflineRL\\source')
from matplotlib.colors import Normalize
import numpy as np
import glob
from offline_ds_evaluation.metrics_manager import MetricsManager
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from utils_gp import interquartile_mean
from scipy.stats import ttest_ind
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests as mult_test
from scipy.stats import levene

from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
 n1, n2 = len(d1), len(d2)
 s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
 s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) #pooling the standard deviations
 u1, u2 = mean(d1), mean(d2)# the means of the samples
 return (u1 - u2) / s # calculate the effect size


with open('params_gp.json', 'r') as f:
    params = json.load(f)

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

envs = {
    'CartPole-v1': 0,
        'MountainCar-v0': 1
        }
params_ex= params["experiment_params"]
experiments = ["ex1","ex2"]
agent_types = params_ex["agent_types"]
algos = agent_types
markers = ["o", "D", "*", "<", "s"]
datasets = [Line2D([0], [0], color="black", marker=markers[i], linewidth=0) for i in range(len(markers))]
useruns = 1
buffer = {"random": "Random", "mixed": "Mixed", "er": "Replay", "noisy": "Noisy", "fully": "Expert"}
modes = list(buffer.keys())
buffer_types =  modes
image_type = "pdf"#"png"

mm = MetricsManager(0)
mm_GP = MetricsManager(0)

mm_dict = {"DQN": mm, "GP": mm_GP}
metman_dict = {mm_GP:["exgp_one_cart", "exgp_one_moun"]}

# Get experiment data -> RL
for ex in experiments:
    for userun in range(1, 6):
        paths = glob.glob(os.path.join("data", ex, f"metrics_*_run{userun}.pkl"))
        for path in paths:
            with open(path, "rb") as f:
                m = pickle.load(f)
                m.recode(userun)
                print(path)
            mm.data.update(m.data)

# Get experiment data -> GP
for metman, experiments in metman_dict.items():
    for ex in experiments:
        for userun in range(1, 6):
            paths = glob.glob(os.path.join("data", ex, f"metrics_*_run{userun}.pkl"))
            for path in paths:
                with open(path, "rb") as f:
                    m = pickle.load(f)
                    m.recode(userun)
                    print(path)
                metman.data.update(m.data)

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
    gp_bool = "GP" if name.split("_")[0] == "GP" else "DQN"

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
    if not data[env][userun][mode].keys() or algo not in data[env][userun][mode].keys():
        data[env][userun][mode][algo] = dict()

    data[env][userun][mode][algo][gp_bool] = csv

def learning_curves():
    for envid, _ in envs.items():
        if envid == "CartPole-v1":
            gp_ex = "gp_one_cart"
            experiment = 1
            y_limit = [0,510]
        elif envid == "MountainCar-v0":
            gp_ex = "gp_one_moun"
            experiment = 2
            y_limit = [-210,-70]

        for buffer_type in buffer_types:
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            fig.subplots_adjust(hspace=0.5)

            fig_iqm, axs_iqm = plt.subplots(2, 2, figsize=(15, 12))
            fig_iqm.subplots_adjust(hspace=0.5)

            for i, agent_type in enumerate(agent_types):
                file_path = fr"C:\Users\david\Dropbox\results\offline_learning\ex{experiment}\{envid}_{agent_type}_{buffer_type}_run{useruns}.xlsx"
                file_path2 = fr"C:\Users\david\Dropbox\results\offline_learning\ex{gp_ex}\{envid}_{agent_type}_{buffer_type}_run{useruns}.xlsx"

                row = i // 2
                col = i % 2
                ax, ax_iqm = axs[row, col], axs_iqm[row, col]

                df = pd.read_excel(file_path)
                #add a running mean column
                df['running_iqm'] = df['iqm'].rolling(window=15).mean()
                df2 = pd.read_excel(file_path2)
                #add a running mean column
                df2['running_iqm'] = df2['iqm'].rolling(window=15).mean()


                # MEAN + SD
                #ax.plot(df["iteration"], df['mean_reward'], label="DQN")
                #ax.plot(df2["iteration"], df2['mean_reward'], label="GP")
                ax.plot(df["iteration"], df['running_iqm'], color="#39568CFF", label="DQN")
                ax.plot(df2["iteration"], df2['running_iqm'], color="#29AF7FFF", label="GP")

                #ax.fill_between(df["iteration"], df['mean_reward'] - df['std_reward'], df['mean_reward'] + df['std_reward'], alpha=0.3)
                #ax.fill_between(df2["iteration"], df2['mean_reward'] - df2['std_reward'], df2['mean_reward'] + df2['std_reward'], alpha=0.3)
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Interquartile Mean Reward')
                ax.set_title(f'{agent_type}')
                ax.legend()
                ax.set_ylim(y_limit)

                # Trendlines
                dqn_trend = np.polyfit(df["iteration"], df['iqm'], 1)
                gp_trend = np.polyfit(df2["iteration"], df2['iqm'], 1)
                ax.plot(df["iteration"], np.polyval(dqn_trend, df["iteration"]), linestyle='--', color="#39568CFF", label='DQN Trend')
                ax.plot(df2["iteration"], np.polyval(gp_trend, df2["iteration"]), linestyle='--', color="#29AF7FFF", label='GP Trend')

                # IQM graph
                # ax_iqm.plot(df["iteration"], df['iqm'], label="DQN")
                # ax_iqm.plot(df2["iteration"], df2['iqm'], label="GP")
                # ax_iqm.set_xlabel('Training Steps')
                # ax_iqm.set_ylabel('Mean Reward')
                # ax_iqm.set_title(f'{agent_type}')
                # ax_iqm.legend()
                # ax_iqm.set_ylim(y_limit)

                # Trendlines
                # dqn_trend_iqm = np.polyfit(df["iteration"], df['iqm'], 1)
                # gp_trend_iqm = np.polyfit(df2["iteration"], df2['iqm'], 1)
                # ax_iqm.plot(df["iteration"], np.polyval(dqn_trend_iqm, df["iteration"]), linestyle='--', color='blue', label='DQN Trend')
                # ax_iqm.plot(df2["iteration"], np.polyval(gp_trend_iqm, df2["iteration"]), linestyle='--', color='orange', label='GP Trend')

            fig.suptitle(f'{envid} - {buffer_type}')
            fig.savefig(fr"C:\Users\david\Dropbox\results\figures_rq_three\learning_curves\Mean\{envid}_{buffer_type}.png")

            #fig_iqm.suptitle(f'{envid} - {buffer_type} - IQM')
            #fig_iqm.savefig(fr"C:\Users\david\Dropbox\results\figures_rq_three\learning_curves\IQM\{envid}_{buffer_type}.png")



def offl_perf_tq_saco():
    outdir = os.path.join("..", "..", "results", "figures_rq_three", "tq_sac_perf")
    os.makedirs(outdir, exist_ok=True)

    c = ["seagreen", "darkcyan", ""]#["red", "tomato", "lightsalmon", "wheat", "palegreen", "limegreen", "green"]
    v = [i / (len(c) - 1) for i in range(len(c))]
    print(v)
    l = list(zip(v, c))
    cmap = "viridis"  #LinearSegmentedColormap.from_list('grnylw',l, N=256)
    normalize = Normalize(vmin=0, vmax=120, clip=True)

    offset_ann = 0.025

    # titles
    x_label = r"SACo of Dataset"
    y_label = r"TQ of Dataset"

    # plot for discussion
    ### algos not averaged

    types = ["all", "noMinAtar", "MinAtar"]
    log_list=[
            ""
          #, "_log"
          ]

    for collector, manager in mm_dict.items():
        for l, log in enumerate(log_list):
            for t, environments in enumerate([list(envs)]):
                if t == 0:
                    f, axs = plt.subplots(2, 2, figsize=(figsize_thesmallplot[0], figsize_thesmallplot[1]), sharex=True, sharey=True)
                    axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]
                else:
                    f, axs = plt.subplots(3, 3, figsize=(figsize_theplot[0], figsize_theplot[1]), sharex=True, sharey=True)
                    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]

                for a, algo in enumerate(algos):
                    ax = axs[a]

                    ax.axhline(y=1, color="silver")
                    ax.axvline(x=1, color="silver")
                    ax.set_title(algo, fontsize="large")

                    for e, env in enumerate(list(environments)):
                        for userun in range(1, 2):
                            online_return = np.max(data[env][1]["online"]["DQN"]["DQN"])
                            random_return = mm.get_data(env, "random", 1)[0][0]
                            online_usap = mm.get_data(env, "er", 1)[2]
                            print(online_return, random_return, online_usap)

                            for m, mode in enumerate(modes):
                                try:
                                    performance = (np.max(np.mean(data[env][userun][mode][algo][collector], axis=1)) - random_return) / (
                                            online_return - random_return) * 100
                                    if log == "_log":
                                        x = np.log(manager.get_data(env, mode, userun)[2]) / np.log(online_usap)
                                    else:
                                        x = manager.get_data(env, mode, userun)[2] / online_usap
                                    y = (manager.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return)
                                except:
                                    continue
                                ax.scatter(x, y, s = 70, c=performance, cmap=cmap, norm=normalize, zorder=10, marker=markers[m])

                f.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs, anchor=(1.3, 0.55),
                    shrink=0.5 if t < 2 else 0.5).set_label(label="Performance in % of Online Policy", size=14)
                f.tight_layout(rect=(0.022, 0.022, 0.87, 1))
                f.legend(datasets, [buffer[m] for m in modes], loc="upper right", bbox_to_anchor=(1, 0.97))
                f.text(0.5, 0.01, x_label if log == "" else "lSACo of Dataset", ha='center', fontsize="large")
                f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
                plt.savefig(os.path.join(outdir, f"algos{log}_{types[t]}_{collector}." + image_type))
                plt.close()

def final_performance():
    #Visualization for the final Performances
    depl_exp = ["ex1", "ex2", "exgp_one_cart", "exgp_one_moun"]

    df_final = pd.DataFrame(columns=["Rewards", "env", "mode", "offline_alg", "coll_alg"])

    for exp in depl_exp:

        if exp == "ex1" or exp == "exgp_one_cart":
            env = "CartPole-v1"
            if exp == "ex1":
                collector = "DQN"
            else:
                collector = "GP"
        elif exp == "ex2" or exp == "exgp_one_moun":
            env = "MountainCar-v0"
            if exp == "ex2":
                collector = "DQN"
            else:
                collector = "GP"

        for buffer_type in buffer_types:
            for agent_type in agent_types:
                file_path = fr"C:\Users\david\Dropbox\results\offline_learning\deployment\{exp}\{env}_{agent_type}_{buffer_type}_run1_final.xlsx"
                df = pd.read_excel(file_path)
                df = df.drop(columns=["Unnamed: 0"])
                df["env"] = env
                df["mode"] = buffer_type
                df["offline_alg"] = agent_type
                df["coll_alg"] = collector
                df_final = pd.concat([df_final, df])

    # Save df final to an excel file
    df_final.to_excel(r'C:\Users\david\Dropbox\results\final_results.xlsx', index = False)
    df_final_agg = df_final.groupby(["env", "mode", "offline_alg", "coll_alg"])["Rewards"].agg(["mean", "std", interquartile_mean]).reset_index()
    df_final_agg.to_excel(r'C:\Users\david\Dropbox\results\final_results_agg.xlsx', index = False)

    # Statistical Tests determining if  the GP is better than the DQN
    #remove the observations with the mode random
    df_final = df_final[df_final["mode"] != "random"]
    buffer_types.remove("random")
    alpha = 0.025


    for test in ["two-sided", "greater"]:
        t_test_dict = dict()
        for env in ["CartPole-v1", "MountainCar-v0"]:
            for mode in buffer_types:
                for agent in agent_types:
                    dqn = df_final[(df_final["env"] == env) & (df_final["mode"] == mode) & (df_final["offline_alg"] == agent) & (df_final["coll_alg"] == "DQN")]["Rewards"].values
                    gp = df_final[(df_final["env"] == env) & (df_final["mode"] == mode) & (df_final["offline_alg"] == agent) & (df_final["coll_alg"] == "GP")]["Rewards"].values

                    cohens_d= abs(cohend(dqn, gp))

                    stat_dqn, p_dqn = stats.shapiro(dqn)
                    stat_gp, p_gp = stats.shapiro(gp)
                    #print("DQN Normal Distribution:", p_dqn > alpha )
                    #print("GP Normal Distribution:", p_gp > alpha )

                    statistic, p_value_lev = levene(dqn, gp)

                    # Check the result
                    # if p_value_lev > alpha:
                    #     print("Fail to reject the null hypothesis. Variances are equal.")
                    # else:
                    #     print("Reject the null hypothesis. Variances are not equal.")

                    t, p = ttest_ind(list(gp), list(dqn), equal_var=False, alternative=test)
                    # if p < alpha:
                    #     print(f"Env: {env}, Mode: {mode}, Agent: {agent}, t: {t}, p: {p}", "Success: GP is better than DQN")
                    # else:
                    #     print(f"Env: {env}, Mode: {mode}, Agent: {agent}, t: {t}, p: {p}", "Fail")
                    t_test_dict[(env, mode, agent)] = [t, p, p_value_lev, p_dqn, p_gp, cohens_d]

        df_t_test = pd.DataFrame.from_dict(t_test_dict, orient='index', columns=["t", "p", "p_lev", "p_dqn", "p_gp", "cohens_d"])
        #sort the dataframe by p ascending
        df_t_test = df_t_test.sort_values(by="p", ascending=True)
        #Bonferroni Correction
        reject, bonf = mult_test(list(df_t_test.p.values), method = "holm", alpha=alpha)[:2]

        df_t_test["Corrected_p"] = bonf
        df_t_test["Reject_H0"] = reject

        print(len(df_t_test[df_t_test["Reject_H0"]==True]))

        #export as excel
        df_t_test.to_excel(fr'C:\Users\david\Dropbox\results\holm_results_{test}.xlsx', index = True)

learning_curves()
#offl_perf_tq_saco()
#final_performance
