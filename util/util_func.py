import matplotlib.pyplot as plt
import scipy.stats
import math
from util.param_set import *
# import library

def get_default_param(argnum):
    '''generate default argument value'''
    return [None] * argnum

def mean_confidence_interval(data, confidence=0.95):
    '''get confidence interval'''
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def truncated_gamma(alpha, beta, truncated_point = 20., size = 1):
    '''function for generating random variable from upper-truncated gamma distribution'''
    while True:
        d = np.random.gamma(alpha, 1. / beta, size = size)  # generate general samples from the gamma distribution
        if (d < truncated_point).all():  # until the samples do not exceed upper bound
            break

    return d

def truncated_normal(mu, std, truncated_point = 20., size = 1):
    '''function for generating random variable from zero-upper-truncated normal distribution'''
    while True:
        d = np.random.normal(mu, std, size = size)  # generate general samples from the normal distribution
        if (d >= 0.).all() and (d < truncated_point).all():  # until the samples do not exceed lower and upper bounds
            break

    return d

def truncated_poisson(lamb, max_d):
    '''function for generating random variable from upper-truncated poisson distribution'''
    prob = np.array([lamb ** i * np.exp(-lamb) / math.factorial(i) for i in range(max_d + 1)])
    prob = prob / np.sum(prob)

    return np.random.choice(range(max_d + 1), p = prob)

def get_demand(alpha, beta, N, demand_dist):
    '''function for simulating demand from predefined distribution'''
    if demand_dist == "gamma":   # gamma demand distribution case
        if N == 1: # single item case
            return truncated_gamma(alpha, beta)
        else:
            return truncated_gamma(alpha, beta, size = N)
    elif demand_dist == "normal":
        if N == 1:
            return truncated_normal(alpha, beta)
        else:
            return truncated_normal(alpha, beta, size = N)
    else:
        if N == 1:
            return truncated_poisson(alpha, beta)
        else:
            pass


def get_reward(a ,_x):
    '''function for computing reward in single inventory system'''
    r = c * a + K + c_b * np.maximum(-_x, 0) + c_h*np.maximum(_x, 0) if a > 0 else c_b*np.maximum(-_x, 0) + c_h*np.maximum(_x, 0)

    return r[0]

def get_multi_reward(a, _x):
    '''function for computing reward in two-item inventory system'''
    r = c_b * np.maximum(-_x, 0).sum() + c_h * np.maximum(_x, 0).sum()
    if a[0] > 0 and a[1] > 0:
        r += c * a.sum() + m * K
    if a[0] > 0 and a[1] == 0:
        r += c * a[0] + K
    if a[0] == 0 and a[1] > 0:
        r += c * a[1] + K

    return r

def get_general_reward(a, _x):
    '''function for computing reward in over-two item (N = 3, 4) inventory system'''
    h = np.where(a > 0., 1., 0.).sum()

    r = c_b * np.maximum(-_x, 0.).sum() + c_h * np.maximum(_x, 0.).sum()
    if h != 0.:
        r += c * a.sum() + h * K * (1 - u * (1 - 1/h))

    return r

def get_average_reward(s, S, horizon, num_seed, demand = None, demand_dist = 'gamma'):
    '''function validation the policy under single invenetory system
    by simulating the long-term horizon and computing average reward value'''
    AR = 0.  # initialize average reward
    seed_record = []

    for seed in range(num_seed):  # iterate many different random seed

        np.random.seed(seed)
        reward_ls = []
        x = np.zeros(1, dtype= np.int32 if demand_dist == "poisson" else np.float64)
        for t in range(horizon):  # iterate predefined long-term horizon
            if x < s:
                a = np.round(S - x[0]) if demand_dist == "poisson"  else S - x[0]
            else:
                a = 0
            d = demand[seed, t]

            _x = x + a - d
            r = get_reward(a, _x)
            reward_ls.append(r)
            x = _x

        mu = np.mean(reward_ls)
        seed_record.append(mu)
        AR += mu

    conf_interval = mean_confidence_interval(seed_record)[-1]
    return AR/num_seed, conf_interval

def get_multi_average_reward(N, s ,c, S, horizon, num_seed, demand = None):
    '''function validation the policy under two-item invenetory system
    by simulating the long-term horizon and computing average reward value'''

    AR = 0.  # initialize average reward
    seed_record = []

    for seed in range(num_seed):  # iterate many different random seed
        np.random.seed(seed)
        reward_ls = []
        x = np.zeros(N)
        for t in range(horizon):  # iterate predefined long-term horizon
            a = np.zeros(N)
            if x[0] < s[0] or x[1] < s[1]:
                if x[0] < c[0]:
                    a[0] = S[0] - x[0]

                if x[1] < c[1]:
                    a[1] = S[1] - x[1]

            d = demand[seed, t]

            _x = x + a - d
            r = get_multi_reward(a, _x)
            reward_ls.append(r)
            x = _x

        mu = np.mean(reward_ls)
        seed_record.append(mu)
        AR += mu

    conf_interval = mean_confidence_interval(seed_record)[-1]
    return AR/num_seed, conf_interval


def get_general_average_reward(N, s, c ,S, horizon, num_seed, demand = None):

    AR = 0.
    seed_record = []

    for seed in range(num_seed):
        np.random.seed(seed)
        reward_ls = []
        x = np.zeros(N)
        for t in range(horizon):
            a = np.zeros(N)
            if (x < s).any():
                a = np.where(x < c, S - x, 0.)

            d = demand[seed, t]

            _x = x + a - d
            r = get_general_reward(a, _x)
            reward_ls.append(r)
            x = _x

        mu = np.mean(reward_ls)
        seed_record.append(mu)
        AR += mu

    conf_interval = mean_confidence_interval(seed_record)[-1]
    return AR / num_seed, conf_interval


def get_adaptive_reward(agent, horizon, num_seed, demand_dist, alg_type = 'SRL-FSA', demand = None):
    '''function validation the policy under single invenetory system with nonstationary scenario
    by simulating the long-term horizon and computing average reward value'''
    from Agent.Single_Agent import Agent
    AR = 0. # initialize average reward
    seed_record = []  # list for saving average cost for random seeds
    hist = np.zeros((horizon, 2)) # array for saving base adaptation trajectory

    for seed in range(num_seed):  # iterate many different random seed
        valid_agent = Agent(1, *Null_param[:-1], stationary = "non-stationary", demand_dist = demand_dist)
        agent.copy_agent(valid_agent)  # construct agent for validation
        np.random.seed(seed)
        reward_ls = []
        x = np.zeros(1)
        valid_agent.t = nonstationary_horizon - adap_interval // 1
        valid_agent.init()

        for t in range(horizon):  # iterate predefined long-term horizon
            if seed == base_seed:
                hist[t,:] = np.array([valid_agent.s, valid_agent.S]).flatten()
            a, eps = valid_agent.get_action(x, stochastic = True)
            d = demand[seed, t: t+1]

            _x = x + a - d
            r = get_reward(a, _x)

            valid_agent.adap_param(d) if t % adap_interval == 0 else valid_agent.adap_param(d, warmup = True)

            valid_agent.update_V(x, r, _x)
            valid_agent.update_policy(x, _x, eps)
            valid_agent.forward()

            reward_ls.append(r)
            x = _x

        if seed == base_seed:  # visualize trajectory of policy parameters under non-stationary scenario with base random seed
            plt.xticks(np.arange(6) * nonstationary_horizon)
            plt.grid()
            plt.plot(hist[:, 0], label = 's')
            plt.plot(hist[:, 1], label = 'S')
            plt.xlabel("$t$")
            plt.ylabel("value")
            plt.legend()
            plt.savefig(f"./result/{alg_type}_non-stationary(N=1).png") # save figure
            plt.close()

        mu = np.mean(reward_ls)
        seed_record.append(mu)
        AR += mu

    conf_interval = mean_confidence_interval(seed_record)[-1]
    return AR/num_seed, conf_interval

def get_multi_adaptive_reward(N, agent, horizon, num_seed, demand_dist, alg_type = 'SRL-FSA', demand = None):
    '''function validation the policy under two-item invenetory system with nonstationary scenario
    by simulating the long-term horizon and computing average reward value'''
    from Agent.Multi_Agent import Agent
    AR = 0. # initialize average reward
    seed_record = [] # list for saving average cost for random seeds
    hist = np.zeros((horizon, 3 * N))  # array for saving base adaptation trajectory

    for seed in range(num_seed):  # iterate on different seeds

        valid_agent = Agent(N, *Null_param, alg_type = alg_type, stationary = "non-stationary", demand_dist = demand_dist)
        agent.copy_agent(valid_agent)  # construct agent for validation
        valid_agent.init()
        np.random.seed(seed)
        reward_ls = []
        x = np.zeros(N)
        valid_agent.t = nonstationary_horizon - adap_interval // N
        valid_agent.init()

        for t in range(horizon):  # iterate predefined long-term horizon
            if seed == base_seed:
                hist[t, :] = np.vstack([valid_agent.s, valid_agent.c, valid_agent.S]).T.flatten()

            a, eps = valid_agent.get_action(x, stochastic = True)
            d = demand[seed, t]
            _x = x + a - d

            valid_agent.adap_param(d) if t % (adap_interval // N) == 0 else valid_agent.adap_param(d, warmup = True)


            r = get_multi_reward(a, _x)

            valid_agent.update_V(x, r, _x)
            valid_agent.update_policy(x, _x, eps)

            reward_ls.append(r)
            valid_agent.forward()
            x = _x

        if seed == base_seed:   # visualize trajectory of policy parameters under non-stationary scenario with base random seed
            plt.grid()
            for n in range(N):
                plt.plot(hist[:, n * 3 + 0], label = f'$s^{n+1}$')
                plt.plot(hist[:, n * 3 + 1], label = f'$c^{n+1}$')
                plt.plot(hist[:, n * 3 + 2], label = f'$S^{n+1}$')

            plt.xlabel("$t$")
            plt.ylabel("value")
            plt.legend()
            plt.savefig(f"./result/{alg_type}_non-stationary(N={N}).png")
            plt.close()

        mu = np.mean(reward_ls)
        seed_record.append(mu)
        AR += mu

    conf_interval = mean_confidence_interval(seed_record)[-1]
    return AR/num_seed, conf_interval


def gen_valid_demand(true_alpha, true_beta, horizon, num_seed, N, demand_dist):
    '''function for generating random demand scenario for several seeds '''
    if N == 1: # for single item case
        val_demand = np.zeros((num_seed, horizon), dtype =np.int32 if demand_dist == 'poisson' else np.float32)

        for seed in range(num_seed):
            np.random.seed(seed)
            for t in range(horizon):
                val_demand[seed, t] = get_demand(true_alpha, true_beta, N, demand_dist)
    else: # for multi-item case
        val_demand = np.zeros((num_seed, horizon, N), dtype=np.float32)

        for seed in range(num_seed):
            np.random.seed(seed)
            for t in range(horizon):
                val_demand[seed, t, :] = get_demand(true_alpha, true_beta, N, demand_dist)

    return val_demand

def gen_nonstationary_demand(horizon, num_seed, N, demand_dist = True):
    '''function for generating random non-stationary scenario for several seeds'''
    if N == 1:  # for single item case
        val_demand = np.zeros((num_seed, len(non_stationary_scenario) * horizon), dtype=np.float32)

        for seed in range(num_seed):
            np.random.seed(seed)
            for p in range(len(non_stationary_scenario)):
                for t in range(horizon):
                    val_demand[seed, t + p * horizon] = get_demand(*non_stationary_scenario[p], N, demand_dist)
    else:
        val_demand = np.zeros((num_seed, len(non_stationary_scenario) * horizon , N), dtype=np.float32)

        for seed in range(num_seed):
            np.random.seed(seed)
            for p in range(len(non_stationary_scenario)):
                for t in range(horizon):
                    val_demand[seed, t + p * horizon, :] = get_demand(*non_stationary_scenario[p], N, demand_dist)

    return val_demand