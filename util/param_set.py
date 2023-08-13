import numpy as np

c = .3  # unit ordering cost
c_h = .1 # unit holding cost
c_b = .5  # unit backlogging cost
K = .1   # fixed cost
m = 1.1  # joint ordering discount term (only for two item)
u = 0.9  # joint ordering discountable ratio

Null_param = [np.nan] * 5  # default argument


dist_param = {'gamma': {'true_alpha': 2.0, 'true_beta': 1.0, 'init_alpha': 3.0, 'init_beta': 2.0},
              'normal': {'true_alpha': 3.0, 'true_beta': 1.0, 'init_alpha': 2.0, 'init_beta': 0.5},
              'poisson': {'true_lambda': 2.0, 'max_d': 7, 'capa': 10}}  # distributional parameter definition

init_s, init_c , init_S = -5., 0., 10. # initial policy parameters
opt_cost = {'gamma': [0.914, 1.788], 'normal': [1.151], 'poisson': [0.903]} # optimal average cost (by full enum or DP)
warmup = 50 # warmup period
base_epoch = 1010 # base period
base_seed = 0
valid_interval = 10  # validation interval
stationary_horizon = 300 # validation horizon for stationary inventory system
stationary_seed = 30  # number of validation random seed for stationary inventory system
nonstationary_horizon = 1000  # validation horizon for a unit regime of non-stationary inventory system
nonstationary_seed = 30  # number of validation random seed for non-stationary inventory system
non_stationary_scenario = [(4.0, 1 / 2 ** .5), (5 / 4., 10 ** .5 / 8), (8., 1.), (1.5, 3 ** .5 / 4.), (16, 2 ** .5)]
# regime-switching distributional parameter scenario for non-stationary case
adap_interval = 100  # initial adaptation inverval for non-stationary case