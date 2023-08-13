from util.util_func import *
import argparse
import pickle
# import library

def get_options():
    '''input description (not used)'''
    optParser = argparse.ArgumentParser()
    optParser.add_argument("-N", type=int, action="store", dest="N", default=1, help="# of items")
    optParser.add_argument("-T", "--type-of-algorithm", action="store", type=str, dest="type",
                         default="SRL-FSA", help="Algoorithm type: SRL-FSA or SRL-PSA")
    optParser.add_argument("-D", "--demand-distribution", dest="demand", default="gamma",
                         action="store", type=str, help="demand distribution: gamma or normal or poisson")
    optParser.add_argument("-S", "--stationary-distribution", dest="stationary", default="stationary",
                         action="store", type=str, help="stationary demand distribution: stationary or non-stationary")
    options = optParser.parse_args()
    return options

def read_inputfile(filename = "./input.txt"):
    '''input by txt file'''
    option_dict = {}
    f = open(filename, 'r')
    while True:
        line = f.readline()
        if not line: break
        key, option = line.strip().split(" ")
        option_dict[key] = option
    f.close()

    return option_dict

if __name__ == "__main__":

    #user_input = input("Enter a value: ")
    #options = get_options()
    dir_PATH = "./input/"
    target_input = "input1.txt"
    option_dict = read_inputfile(dir_PATH + target_input)  # read txt input
    '''just change value of `target_input` as preferred one such as  "input2.txt" or "input3.txt" ... '''
    print(option_dict)

    N = np.int32(option_dict['N'])  # input number of items
    single = False if N > 1 else True  # check whether single item or not
    demand_dist = option_dict['D']   # input type of demand distribution
    alg_type = option_dict['T']  # input type whether partial SA (SRL-PSA) or full SA (SRL-FSA)
    stationary = option_dict['S'] # input mode whether stationary system
    record_AR = []  # list for saving intermediate average cost of policy
    epoch = base_epoch + warmup  # epoch for algorithmic iteration

    if demand_dist == "poisson": # poisson demand distribution case
        true_alpha = lamb = dist_param[demand_dist]['true_lambda'] # assign lambda of poisson
        true_beta = max_d = dist_param[demand_dist]['max_d'] # assign truncated upper bound
        capa = dist_param[demand_dist]['capa'] # assign retailer capacity

    else: # gamma or normal demand cases
        true_alpha = dist_param[demand_dist]['true_alpha'] # assign true alpha (mu) parameter of gamma (normal)
        true_beta = dist_param[demand_dist]['true_beta'] # assign true beta (sigma) parameter of gamma (normal)

        init_alpha = dist_param[demand_dist]['init_alpha'] # assign initial alpha (mu) parameter of gamma (normal)
        init_beta = dist_param[demand_dist]['init_beta'] # assign initial beta (mu) parameter of gamma (normal)


    x = np.zeros(N, dtype = np.int32 if demand_dist == 'poisson' else np.float64)  # initialize inventory level (= 0)

    if single: # single item case
        if demand_dist == 'poisson': # poisson demand distribution case
            from Agent.Discrete_Agent import Agent  # structured RL agent for managing discrete demand
            S = capa
            s = -capa + max_d # initial policy parameter (extreme case)
            agent = Agent(N, s, S, max_d, capa, alg_type= alg_type, demand_dist = demand_dist, stationary = "stationary")
            # define structure RL agent
        else:
            from Agent.Single_Agent import Agent   # structured RL agent for managing single-item continuous-type demand
            S = init_S
            s = init_s # initial policy parameter
            agent = Agent(N, s, S, init_alpha, init_beta, alg_type=alg_type, demand_dist=demand_dist, stationary="stationary")
            # define structure RL agent
        agent.init() # initialize setting of agent

    else: # multiple item case
        if N == 2:
            from Agent.Multi_Agent import Agent  # structured RL agent for managing two-item continuous-type demand
        elif N == 3:
            from Agent.Tri_Agent import Agent # structured RL agent for managing three-item continuous-type demand
        elif N == 4:
            from Agent.Quad_Agent import Agent # structured RL agent for managing four-item continuous-type demand
        S = init_S
        c = init_c
        s = init_s  # initial policy parameter

        agent = Agent(N, s, c, S, init_alpha, init_beta, alg_type= alg_type, demand_dist = demand_dist, stationary = "stationary")
        # define structure RL agent for multiple items
        agent.init() # initialize setting of agent

    val_demand = gen_valid_demand(true_alpha, true_beta, stationary_horizon, stationary_seed, N = N, demand_dist = demand_dist)
    # validation demand trajectory (multiple seeds)
    nonstationary_demand = None

    if stationary != "stationary":
        nonstationary_demand = gen_nonstationary_demand(nonstationary_horizon, nonstationary_seed, N = N, demand_dist = demand_dist)

    np.random.seed(0)
    warmupflag = True

    for i in range(epoch):
        if i == warmup:
            warmupflag = False

        if warmupflag:  # during warmup period
            d = get_demand(alpha=true_alpha, beta=true_beta, N=N, demand_dist=demand_dist)
            agent.adap_param(d, warmup = warmupflag)
        else:
            a, eps = agent.get_action(x, stochastic = True)  # select action by following structured policy
            d = get_demand(alpha=true_alpha, beta=true_beta, N=N, demand_dist=demand_dist) # get demand
            _x = x + a - d  # system dynamics
            agent.adap_param(d, warmup=warmupflag)  # adapt underlying distributional parameter
            r = get_reward(a, _x) if single else get_multi_reward(a, _x) if N == 2 else get_general_reward(a, _x)
            # get reward
            agent.update_V(x, r, _x)  # update value function
            agent.update_policy(x, _x, eps)  # update policy parameter
            x = _x
            agent.forward() # forward next step


    for i in range(epoch - warmup):  # evaluate for every valid interval
        if i % valid_interval == 0:
            if single:
                AR, conf_interval = get_average_reward(agent.record_s[i], agent.record_S[i], stationary_horizon, stationary_seed, val_demand, demand_dist = demand_dist)
            else:
                if N == 2:
                    AR, conf_interval = get_multi_average_reward(N, agent.record_s[i], agent.record_c[i], agent.record_S[i], stationary_horizon, stationary_seed, val_demand)
                else:
                    AR, conf_interval = get_general_average_reward(N, agent.record_s[i], agent.record_c[i], agent.record_S[i], stationary_horizon, stationary_seed, val_demand)
            record_AR.append(AR)  # save average cost result

    print(f"Under {demand_dist}, {alg_type}'s (N={N}) stationary cost: {AR} +/- {conf_interval}")
    if single:
        print(f"optimized (s, S) for item 1: ({agent.s[0]}, {agent.S[0]})")
    else:
        for n in range(N):
            print(f"optimized (s, c, S) for item {n + 1}: ({agent.s[n]}, {agent.c[n]}, {agent.S[n]})")
    # print optimized policy parameters

    if stationary != "stationary": # non-stationary scenario
        if single:
            AR, conf_interval = get_adaptive_reward(agent, len(non_stationary_scenario) * nonstationary_horizon, nonstationary_seed, demand = nonstationary_demand, alg_type= alg_type, demand_dist = demand_dist)
        else:
            AR, conf_interval = get_multi_adaptive_reward(N, agent, len(non_stationary_scenario) * nonstationary_horizon, nonstationary_seed, demand = nonstationary_demand, alg_type= alg_type, demand_dist = demand_dist)
        print(f"Under {demand_dist}, {alg_type}'s (N={N}) nonstationary cost: {AR} +/- {conf_interval}")

    if single:
        plt.plot(agent.record_s, label='s')
        plt.plot(agent.record_S, label='S')
    else:
        for n in range(N):
            plt.plot(np.array(agent.record_s)[:, n], label='$s^{n}$')
            plt.plot(np.array(agent.record_c)[:, n], label='$c^{n}$')
            plt.plot(np.array(agent.record_S)[:, n], label='$S^{n}$')

    plt.xlabel("$t$")
    plt.ylabel("value")

    plt.grid()
    plt.legend()
    plt.savefig(f"./result/{alg_type}_{demand_dist}_trajectory(N={N}).png")
    plt.close()

    plt.xlabel("t")
    plt.ylabel("average cost")
    plt.grid()
    # visualize updated trajectory result

    if N < 3:
        plt.hlines(opt_cost[demand_dist][N - 1], 0, len(record_AR) * valid_interval, label='near-optimal', color='r', linestyles="-.", zorder=10)
    plt.plot(np.arange(len(record_AR)) * valid_interval, record_AR, label = alg_type)
    plt.legend()
    plt.savefig(f"./result/{alg_type}_{demand_dist}_converge_costperf(N={N}).png")
    plt.close()  # save figure for convergence trend

    #pickle.dump(record_AR, open(f"./result/record_{alg_type}_{demand_dist}(N={N}).pkl", "wb"))

