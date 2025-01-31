import os
import shutil
import numpy as np
from tqdm import tqdm

useruns = 5
runs = 5

envs = ['CartPole-v1', "MountainCar-v0"]#, "MiniGrid-LavaGapS7-v0", "MiniGrid-Dynamic-Obstacles-8x8-v0",
        #"Breakout-MinAtar-v0", "SpaceInvaders-MinAtar-v0"]
algos = ["BC",
         #"BVE",
         #"MCE",
         "DQN",
         #"QRDQN",
         "REM",
         #"BCQ",
         "CQL",
         #"CRR"
         ]
datasets = ["random", "mixed", "er", "noisy", "fully"]

#origin = os.path.join("..", "..", "results", "raw")
origin = os.path.join("results", "raw")
target = os.path.join("results", "csv_per_userun")
os.makedirs(target, exist_ok=True)

folders = ["avd", "return"]

### join per userun
for gp_bool in [True, False]:
    for e, env in enumerate(envs):
        if env == "MountainCar-v0":
            envid = "moun"
        elif env == "CartPole-v1":
            envid = "cart"
        for algo in tqdm(algos, desc=f"{env}"):
            for ds in datasets:
                for folder in folders:
                    for userun in range(useruns):
                        results = []
                        os.makedirs(os.path.join(target, folder, env, f"userun{userun + 1}"), exist_ok=True)
                        for run in range(runs):
                            try:
                                if not gp_bool:
                                    results.append(np.genfromtxt(os.path.join(origin, folder, f"ex{e + 1}", f"userun{userun + 1}",
                                                                        f"{env}_{algo}_{ds}_run{run + 1}.csv")).tolist())
                                if gp_bool:
                                    results.append(np.genfromtxt(os.path.join(origin, folder, f"exgp_one_{envid}", f"userun{userun + 1}",
                                                                        f"{env}_{algo}_{ds}_run{run + 1}.csv")).tolist())
                                    print(results)
                            except:
                                print(f"no data available for GP:{gp_bool} {env}, {algo}, {ds}, userun {userun + 1}, run{run + 1}")
                        length = len(results)
                        # if no data available, break out of loop
                        if length == 0:
                            continue
                        results = np.asarray(results).reshape(length, -1).transpose((1, 0))
                        if gp_bool:
                            np.savetxt(os.path.join(target, folder, env, f"userun{userun + 1}", f"GP_{algo}_{ds}.csv"),
                                results, delimiter=";")
                        else:
                            np.savetxt(os.path.join(target, folder, env, f"userun{userun + 1}", f"{algo}_{ds}.csv"),
                                    results, delimiter=";")
                    # online
                    if not gp_bool:
                        for run in range(runs):
                            shutil.copy(os.path.join(origin, folder, f"ex{e + 1}", f"{env}_online_run{run + 1}.csv"),
                                        os.path.join(target, folder, env, f"userun{run + 1}", "DQN_online.csv"))
