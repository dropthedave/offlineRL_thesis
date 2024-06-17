#    This experiment is compatible with SCOOP.
#    After installing scoop you can run the program with python -m scoop basic-gp.py
#    This file uses DEAP (https://github.com/DEAP/deap)
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import numpy
import gym
from gym import wrappers
import operator
import random
import os
import pickle
import math
from tqdm import tqdm

#from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from source.utils.buffer import ReplayBuffer
#from scoop import futures
import copy
from source.offline_ds_evaluation.evaluator import Evaluator
from source.offline_ds_evaluation.metrics_manager import MetricsManager
from source.train_offline import train_offline
import pygraphviz as pgv
import json
import pandas as pd
from utils_gp import interquartile_mean
from utils_gp import eaSimple

#Line 370:
############################################################################
# Operators: Sel: Tournament, Crossover: Semantic, Mutation: Semantic
############################################################################
selstr = "tournament"
cxstr = "semantic"
mutstr = "semantic"

#Experiment Name
experiment = "gp_five"
with open('params_gp.json', 'r') as f:
    params = json.load(f)
parameters = params["experiment_params"]

params_gp = params["params_gp"]

#Experimental parameters
buffer_types = ["random", "mixed", "er", "noisy", "fully"]
envid = "CartPole-v1"
#envid="MountainCar-v0"

#Observation space
env = gym.make(envid)
obs_space = len(env.observation_space.high)
env.close()

#Online transitions
transitions = parameters["transitions_online"]
offline_bool = False
seed = 42
useruns= parameters["runs_online"]

agent_types = parameters["agent_types"]

# hyperparameters for offline training
transitions_offline = transitions #5*transitions
#Batch size
batch_size = parameters["batch_size_offline"]
lr = [1e-4] * len(agent_types)
discount = 0.95
dataset = -1

def progn(*args):
    for arg in args:
        arg()

#Define different variables/hyperparameters for the two environments:
if envid == "MountainCar-v0":
    experiment = experiment + "_moun"
    def wrap(num):
        if num <= -0.5:
            return 0
        if -0.5 < num <= 0.5:
            return 1
        else:
            return 2

elif envid == 'CartPole-v1':
        experiment = experiment + "_cart"
        def wrap(num):
            if num <= 0:
                return 0
            else:
                return 1

os.makedirs(rf'C:\Users\david\Dropbox\results\gp_output\{experiment}', exist_ok=True)
def limit(input, minimum, maximum):
    if input < minimum:
        return minimum
    elif input > maximum:
        return maximum
    else:
        return input

def protdDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1


#add primitive functions to the set
def lf(x): return 1 / (1 + numpy.exp(-x))

pset = gp.PrimitiveSet("MAIN", arity=obs_space) #Arity is the number of inputs
pset.addPrimitive(numpy.add, 2, name="add")
pset.addPrimitive(numpy.subtract, 2, "sub")
pset.addPrimitive(numpy.multiply, 2, name="mul")
pset.addPrimitive(protdDiv, 2) # protected division -> if division fails -> return 1
pset.addPrimitive(lf, 1, name="lf") # For semantic crossovers
pset.addPrimitive(numpy.sign, 1, name="sgn") # numpy.sign -> returns the sign of a number (1 for positive, -1 for negative, 0 for zero)
pset.addPrimitive(numpy.sin, 1, name="sen") # calculates the sine of a number
pset.addEphemeralConstant("R", lambda: round(numpy.random.uniform(low=-1, high=1), 2))


creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Makes sure the direction is right -> fitness should be maximized
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax) #Creates a "individual" class which gets the tree and the direction given

if envid == "MountainCar-v0":
    pset.renameArguments(ARG0="CarPos")
    pset.renameArguments(ARG1="CarVel")

if envid == 'CartPole-v1':
    pset.renameArguments(ARG0="CartPos")
    pset.renameArguments(ARG1="CartVel")
    pset.renameArguments(ARG2="PolAng")
    pset.renameArguments(ARG3="PolVel")

toolbox = base.Toolbox()

def graph(expr, run):
    nodes, edges, labels = gp.graph(expr)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
        n.attr["fontsize"] = "7"
        n.attr["width"] = "0.9"
    g.draw(rf'C:\Users\david\Dropbox\results\gp_output\{experiment}\{experiment}-{run}-tree.png')

#####################################
# Fitness Function
#####################################
def evalIndividual(individual, render=False):
    iqm_list = []

    if render:
        # create a temporary variable with our env, which will use rgb_array as render mode. This mode is supported by the RecordVideo-Wrapper
        #tmp_env = gym.make(envid, render_mode="rgb_array")

        # wrap the env in the record video
        #env = gym.wrappers.RecordVideo(env=tmp_env, video_folder=rf"C:\Users\david\Dropbox\results\{experiment}-video", name_prefix=f"{experiment}-{envid}-video", episode_trigger=lambda x: x % 1 == 0)
        env = gym.make(envid)
        n_evals = 10
    else:
        env = gym.make(envid)
        n_evals = 1

    state, info = env.reset()
    #if render:
        #env.start_video_recorder()

    # Transform the tree expression to functional Python code
    action_fun = gp.compile(individual, pset)
    # graph
    # print(action)
    fitness = 0
    failed = False
    counter_steps = 0
    done = False
    for x in range(0, n_evals):
        state, info = env.reset()
        done = False
        try:
            iqm_list.append(fitness_iqm)
        except:
            fitness_iqm = 0
        fitness_iqm = 0
        while not done:
            action_result = action_fun(*state)
            action=wrap(action_result)
            new_state, reward, done, truncated, info = env.step(action)
            if truncated:
                done = True
            er_buffer.add(state, action, reward, done)
            state = new_state
            # if(fitness > 450):
            #     env.render()
            fitness += reward
            fitness_iqm += reward
            counter_steps += 1

    if render:
        #env.close_video_recorder()
        avg_fitness_el = fitness/n_evals
        iqm_list.append(fitness_iqm)
        iqm = interquartile_mean(iqm_list)
        return avg_fitness_el, iqm

    return (-2000,) if failed else (fitness,)

#####################################
# generate transitions from trained agent
#####################################
def final_policy_collection(individual, run):
    env = gym.make(envid)
    action_fun = gp.compile(individual, pset)
    done = True

    for i in  tqdm(range(transitions), desc=f"Collect expert data ({envid}), run {run}"):
        if done:
            state, info = env.reset()
        action_result = action_fun(*state)
        action=wrap(action_result)
        next_state, reward, done, truncated, info = env.step(action)
        if truncated:
            done = True
        final_policy_buffer.add(state, action, reward, done)
        state = next_state

    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_fully.pkl"), "wb") as f:
        pickle.dump(final_policy_buffer, f)

#####################################
# generate noisy transitions from trained agent
#####################################
def final_policy_noisy_collection(individual, run):
    env = gym.make(envid)
    action_fun = gp.compile(individual, pset)
    done = True
    n_actions = env.action_space.n

    for i in tqdm(range(transitions), desc=f"Collect noisy data ({envid}), run {run}"):
        if done:
            state, info = env.reset()
        if numpy.random.uniform(low=0, high=1) < 0.2:
            action = numpy.random.randint(0,n_actions-1)
        else:
            action_result = action_fun(*state)
            action=wrap(action_result)
        next_state, reward, done, truncated, info = env.step(action)
        if truncated:
            done = True
        noisy_policy_buffer.add(state, action, reward, done)
        state = next_state

    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_noisy.pkl"), "wb") as f:
        pickle.dump(noisy_policy_buffer, f)

#####################################
# generate random transitions
#####################################
def random_collector(run):
    done, rng, n_actions = True, numpy.random.default_rng(seed), env.action_space.n
    for _ in tqdm(range(transitions), desc=f"Evaluate random policy ({envid}), run {run}"):
        if done:
            state, info = env.reset()#seed=seed)

        action = rng.integers(n_actions)
        #next_state, reward, done, _ = env.step(action)
        result = env.step(action)
        next_state = result[0]
        reward = result[1]
        done = result[2]
        truncated = result[3]
        if done or truncated:
            done = True

        random_buffer.add(state, action, reward, done)

        state = next_state

    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_random.pkl"), "wb") as f:
        pickle.dump(random_buffer, f)

    #####################################
    # generate mixed transitions (random + fully)
    #####################################
    random_buffer.mix(buffer=final_policy_buffer, p_orig=0.8)
    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_mixed.pkl"), "wb") as f:
        pickle.dump(random_buffer, f)

def assess_env(run):
    use_run = run
    env = envid

    with open(os.path.join("data", f"ex{experiment}", f"{env}_run{use_run}_er.pkl"), "rb") as f:
        buffer = pickle.load(f)
    state_limits = []
    for axis in range(len(buffer.state[0])):
        state_limits.append(numpy.min(buffer.state[:, axis]))
        state_limits.append(numpy.max(buffer.state[:, axis]))
    action_limits = copy.deepcopy(state_limits)
    action_limits.append(numpy.min(buffer.action))
    action_limits.append(numpy.max(buffer.action))

    results, returns, actions, entropies, sparsities, episode_lengts = [], [], [], [], [], []
    mm = MetricsManager(experiment)

    for buffer_type in buffer_types:
        with open(os.path.join("data", f"ex{experiment}", f"{env}_run{use_run}_{buffer_type}.pkl"), "rb") as f:
            buffer = pickle.load(f)

        evaluator = Evaluator(env, buffer_type, buffer.state, buffer.action, buffer.reward,
                              numpy.invert(buffer.not_done))

        if env == 'CartPole-v1' or env == 'MountainCar-v0':
            rets, us, usa, ents, sps, epls = evaluator.evaluate(state_limits=state_limits, action_limits=action_limits,
                                                                epochs=10)
            #the return: returns, unique_states, unique_state_actions, entropies, sparsities, ep_lengths
        else:
            rets, us, usa, ents, sps, epls = evaluator.evaluate(epochs=10)

        returns.append(rets)
        entropies.append(ents)
        sparsities.append(sps)
        episode_lengts.append(epls)
        actions.append(buffer.action.flatten().tolist())

        results.append([env, buffer_type, (numpy.mean(rets), numpy.std(rets)), usa, (numpy.mean(ents), numpy.std(ents))])

        mm.append([env, buffer_type, (numpy.mean(rets), numpy.std(rets)), us, usa, (numpy.mean(ents), numpy.std(ents)),
                   (numpy.mean(sps), numpy.std(sps)), (numpy.mean(epls), numpy.std(epls))])

    with open(os.path.join("data", f"ex{experiment}", f"metrics_{env}-GP_run{use_run}.pkl"), "wb") as f:
        pickle.dump(mm, f)

def train(use_run, run=1, dataset=-1):
    env = envid
    for a, agent in enumerate(agent_types):
        for bt in range(len(buffer_types)):
            if 0 < dataset != bt:
                continue

            train_offline(experiment=experiment, envid=env, agent_type=agent, buffer_type=buffer_types[bt],
                          discount=discount, transitions=transitions_offline, batch_size=batch_size, lr=lr[a],
                          use_run=use_run, run=run, seed=seed+run, use_remaining_reward=(agent == "MCE"))


#Genetic Operators!
# Attribute generator -> Initialisation method
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=5)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init) #create new individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #create new population based on toolbox. init

#Evaluation function
toolbox.register("evaluate", evalIndividual)

#Parents selection
if selstr == "tournament":
    toolbox.register("select", tools.selTournament, tournsize=params_gp["tournsize"])
elif selstr == "doubletournament":
    toolbox.register("select", tools.selDoubleTournament, fitness_size=params_gp["tournsize"], parsimony_size=2, fitness_first = True)

#Crossover
if cxstr == "onepoint":
    toolbox.register("mate", gp.cxOnePoint)
elif cxstr == "semantic":
    toolbox.register("mate", gp.cxSemantic, pset = pset, min=params_gp["mut_min"],max=params_gp["mut_max"])

#Mutation
if mutstr == "semantic":
    toolbox.register("mutate", gp.mutSemantic, pset=pset, min = params_gp["mut_min"], max= params_gp["mut_max"], ms=params_gp["ms"])
elif mutstr== "uniform":
    toolbox.register("expr_mut", gp.genGrow, min_=params_gp["mut_min"], max_=params_gp["mut_max"]) #wie groß können die Mutations ausfallen
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=params_gp["maxsize"]))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=params_gp["maxsize"]))

def main():
    run_list = []
    size_list = []
    fit_list = []
    iqm_list = []

    for run in range(1, useruns+1):
        if not offline_bool:
            #Get Buffers
            global er_buffer
            er_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
            global final_policy_buffer
            final_policy_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
            global noisy_policy_buffer
            noisy_policy_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
            global random_buffer
            random_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
            #Set population size
            pop = toolbox.population(n=params_gp["n_pop"])
            #Set size for elitism
            hof = tools.HallOfFame(1)

            #Set the metrics and statistics to collect
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", numpy.mean)
            mstats.register("std", numpy.std)
            mstats.register("min", numpy.min)
            mstats.register("max", numpy.max)

            #Evolutionary process:
            ##########################################################################################
            pop, log = eaSimple(pop, toolbox, cxpb = params_gp["cxpb"], mutpb=params_gp["mutpb"],
                                ngen= params_gp["ngen"], replay_memory= er_buffer,
                                transitions=transitions, stats=mstats,
                                halloffame=hof, verbose=True)
            ##########################################################################################

            #Save the collected trajectories
            os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
            with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_er.pkl"), "wb") as f:
                pickle.dump(er_buffer, f)

            #winner = gp.compile(hof[0], pset)
            graph(hof[0], run=run)

            # Collect trajectories with the elite policy
            final_policy_collection(hof[0], run)
            # Collect trajectories noisy
            final_policy_noisy_collection(hof[0], run)
            # Collect trajectories randomly
            random_collector(run)

            #Assess the collected datasets:
            assess_env(run)

            #Evaluation of Individual (Video)
            avg_fitness_el, iqm=evalIndividual(hof[0], True)

            # Save the stats of hof as a json
            stats = mstats.compile(hof)

            elite_size = int(stats["size"]["avg"])
            elite_avg_fit = avg_fitness_el
            iqm_list.append(iqm)
            run_list.append(run)
            size_list.append(elite_size)
            fit_list.append(elite_avg_fit)

            #hier df draus machen und dann als excel speichern -> einfacher zu verarbeiten.

        #Train offline RL algorithms:
        elif offline_bool:
            train(use_run=run, run=1)

    df = pd.DataFrame({'run': run_list,
                               'elite_size': size_list,
                               'elite_avg_fit': fit_list,
                               "elite_iqm_fit":iqm_list})

    df.to_excel(rf"C:\Users\david\Dropbox\results\gp_output\{experiment}\{experiment}-stats.xlsx", index=False)
    # print log
    return pop, log, hof


if __name__ == "__main__":
    main()
