import numpy as np
from collections import deque
from Agent.util.hyper_param_set import *
# import library
np.random.seed(0)

class Agent:
    """
    Structured RL for Single-item inventory system

    :param N: (int) # of items
    :param s: (float) initial reorder level
    :param S: (float) initial order-up-to level
    :param alpha: (float) initial estimated distributional parameter (first parameter)
    :param beta: (float) initial estimated distributional parameter (second parameter)
    :param alg_type: (string) assign type whether partial SA (SRL-PSA) or full SA (SRL-FSA)
    :param stationary: (string) assign mode whether stationary system (stationary) or not (non-stationary)
    :param demand_dist: (string) assign type of demand distribution (gamma, normal, poisson)
    """
    def __init__(self, N, s, S,  alpha, beta, alg_type = "SRL-FSA", stationary = "stationary", demand_dist = "gamma"):

        self.N = N  # assign number of items
        self.s = s * np.ones(N)  # initialize reorder level
        self.S = S * np.ones(N)  # initialize order-up-to level
        self._s = self.s.copy()
        self._S = self.S.copy()

        self.demand_dist = demand_dist  # assign distributional type mode
        self.h = h   # normalize factor for input on the relative value function
        self.cnt = deque(maxlen = max_len)  # define the observation queue

        self.alg_type = alg_type  # assign type of algorithm
        self.prev_alpha = self.alpha = alpha  # estimated scale parameter for gamma distribution
        self.prev_beta = self.beta = beta   # estimated shape parameter for gamma distribution
        self.stationary = stationary    # assign mode whether stationary system or not

        if stationary == "stationary":
            self.init_tau = self.tau = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['tau']
            # initialize tau as a hyper-parameter of sigmoid function
            self.init_sigma = self.sigma = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['sigma']
            # initialize sigma for standard deviation of noise distribution in stochastic policy
            self.mt = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['mt']
            # momentum update factor
        else:
            self.tau = self.init_tau = 0.
            self.init_sigma = self.sigma = 0.

        self.record_s = [self.s]
        self.record_S = [self.S]
        # record queue for tracking the trajectories for each policy parameter

        self.dim = 4  # dimension for relative value function (=4 for 4th order polynomial regression)
        self.gS = self.gs = 0. #  initialize momentum variable
        self.w = np.zeros(self.dim)    # initialize parameter for relative value function
        self.w[-1] = low_pos   # initialize the parameter for highest order term with small positive value

        self.t = 0    # initialize time index
        self.rho = 0.    # initialize relative value
        self.forward()   # conduct initial forward step

    def init(self):
        '''function for initializing additional setting'''
        if self.alg_type == 'SRL-FSA':
            self.m = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['m']
            #  size of batch sample for policy update using full SA (FSA)

    def get_action(self, x, stochastic = True):
        '''function for producing action by following the policy'''
        noise = None   # initialize noise term

        if stochastic:  # if stochastic (s,S) replenishment mode
            p = self.f(x)  # assign sigmoid mixing probability on current state x
            u = np.random.rand(1)  # sampling from uniform
            noise_S = np.random.normal(loc = self.S, scale = self.sigma)  #  sampling from normal (S, sigma)
            noise = noise_S - self.S
            if p < u:
                a = np.maximum(noise_S - x, 0.)  # if negative value then zero clipping
            else:
                a = 0
        else:    # if deterministic (s,S) replenishment mode
            if x < self.s:
                a = self.S - x
            else:
                a = 0

        return a, noise

    def poly_basis(self, x):
        '''function for computing polynomial basis'''
        return np.array([(x / self.h) ** (i+1) for i in range(self.dim)], dtype=np.float32)


    def V(self, x):
        '''function for computing relative value'''
        pi = self.poly_basis(x)
        valuef = np.dot(self.w, pi)

        return valuef

    def update_V(self, x, r, _x):
        '''function for updating relative value function'''
        TD_target = r + self.V(_x[0]) - self.rho
        self.rho += self.a * (r + self.V(_x[0]) - self.V(x[0]) - self.rho)
        self.w += self.eta * (TD_target - self.V(x[0])) * self.poly_basis(x[0])
        self.w[-1] = np.maximum(0., self.w[-1])    # restrict to highest order term as nonnegative value

    def f(self, x):
        '''sigmoid function in point x given s and tau value'''
        return 1. / (1 + np.exp(-(x - self.s) / self.tau))

    def dfds(self, x):
        '''function for computing the derivative value of the sigmoid function with respect to reorder level'''
        sig = self.f(x)

        return sig * (sig - 1) / self.tau

    def get_moment(self, order, alpha):
        '''function for computing the n-th moment for gamma distribution (order <= 4)'''
        if order ==1:
            mnt = alpha / self.beta
        elif order == 2:
            mnt = alpha * (alpha + 1) / self.beta ** 2
        elif order == 3:
            mnt = alpha * (alpha + 1) * (alpha + 2) / self.beta ** 3
        else:
            mnt = alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) / self.beta ** 4

        return mnt

    def get_exp_diff(self, idx, alpha, x):
        '''function for computing expectation for the the n-th equation of the gamma random variable (order <= 4)'''
        order = idx + 1

        if order == 1:
            res = x ** order - self.get_moment(order, alpha)
        elif order == 2:
            res = x ** order - 2 * x ** (order - 1) * self.get_moment(order - 1, alpha) + self.get_moment(order, alpha)
        elif order == 3:
            res = x ** order - 3 * x ** (order - 1) * self.get_moment(order - 2, alpha) + 3 * x ** (order - 2) * self.get_moment(order - 1, alpha) - self.get_moment(order, alpha)
        else:
            res = x ** order - 4 * x ** (order - 1) * self.get_moment(order - 3, alpha) + 6 * x ** (order - 2) * self.get_moment(order - 2, alpha) - 4 * x ** (order - 3) * self.get_moment(order - 1, alpha) + self.get_moment(order, alpha)

        return res

    def adap_param(self, obs, warmup = False):
        '''function for adapting the estimation of the underlying parameters of gamma distribution'''
        self.cnt.append(obs)
        n = len(self.cnt)
        if not warmup:
            sum_obs = np.sum(self.cnt)
            log_dev = np.log(sum_obs / n) - np.sum(np.log(self.cnt)) / n

            self.alpha = np.maximum((3. - log_dev + np.sqrt((log_dev - 3.) ** 2 + 24. * log_dev)) / 12. / log_dev, 1.1)
            # maximum likelihood estimator for alpha of underlying dist
            self.beta = n * self.alpha / sum_obs  # maximum likelihood estimator for beta of underlying dist


    def get_P0(self, x, size):
        '''function for generating random sample from the distribution P0
        which is the transition probability under inventory exceeding reorder level '''
        sample_x = x - np.random.gamma(self.alpha, 1 / self.beta, size)

        return sample_x

    def get_P1(self, size, alpha, eps):
        '''function for generating random sample from the distribution P1
        which is the transition probability under inventory being below reorder level '''
        sample_x = self.S + eps - np.random.gamma(alpha, 1 / self.beta, size)

        return sample_x

    def get_PbarS(self, ber, eps, size):
        '''function for generating random sample from the composite distribution two different parameterized P1
        which is distinguished by random variable from the bernoulli, represented as follow (1- B) * P1(x;a-1) + B * P1(x;a)'''

        sample_x = np.zeros(size)
        for i in range(size):
            sample_x[i] = self.get_P1(1, self.alpha, eps) if ber[i] else self.get_P1(1, self.alpha - 1, eps)

        return sample_x

    def get_Pbars(self, x, ber, eps, size):
        '''function for generating random sample from the composite distribution two different parameterized P1
        which is distinguished by random variable from the bernoulli, represented as follow (1- B) * P0(x;a) + B * P1(x;a)'''

        sample_x = np.zeros(size)
        for i in range(size):
            sample_x[i] = self.get_P1(1, self.alpha, eps) if ber[i] else self.get_P0(x, 1)

        return sample_x

    def project_s(self, x, S):
        '''projection operator to restrict the following equation s =< S'''
        return np.minimum(x, S)

    def update_policy(self, x, _x, eps):
        '''function for updating the policy parameters'''

        if self.alg_type == 'SRL-PSA':  # if partial stochastic approximation (PSA) based algorithm

            self.gS = self.mt * self.gS + (1 - self.mt) * (1 - self.f(x)) * self.beta * np.sum([self.w[i] * (self.get_exp_diff(idx = i, alpha =self.alpha - 1, x = self.S + eps)
                                                                                                             - self.get_exp_diff(idx = i, alpha =self.alpha, x = self.S + eps)) for i in range(self.dim)])
            _S = self.S - self.b1 * self.gS  # update formula for order-up-to level

            self.gs = self.mt * self.gs + (1 - self.mt) * self.dfds(x) * np.sum([self.w[i] * (self.get_exp_diff(idx = i, alpha =self.alpha, x = x)
                                                                                              - self.get_exp_diff(idx = i, alpha =self.alpha, x = self.S + eps)) for i in range(self.dim)])
            self.s = self.project_s(self.s - self.b2 * self.gs, _S)  # update formula for reorder level
            self.S = _S
        else:   # if full stochastic approximation (FSA) based algorithm

            yS_hat = np.random.gamma(self.alpha - 1, 1 / self.beta, size = self.m)
            yS = np.random.gamma(self.alpha, 1 / self.beta, size= self.m)  # sampling from underlying distribution for gradient approximation of order-up-to level
            self.gS = self.mt * self.gS + (1 - self.mt) * np.clip(self.beta * (1 - self.f(x[0])) * np.sum([self.w[i] * (np.mean((self.S + eps - yS_hat) ** (i + 1))
                                                                                                                        - np.mean((self.S + eps - yS) ** (i + 1))) for i in range(self.dim)]), -bigM, bigM)
            _S = self.S - self.b1 * self.gS  # update formula for order-up-to level

            ys = np.random.gamma(self.alpha, 1 / self.beta, size=self.m)  # sampling from underlying distribution for gradient approximation of reorder level
            self.gs = self.mt * self.gs + (1 - self.mt) * np.clip(self.dfds(x[0]) * np.sum([self.w[i] * (np.mean((x - ys) ** (i + 1))
                                                                                                         - np.mean((self.S + eps - ys) ** (i + 1))) for i in range(self.dim)]), -bigM, bigM)
            self.s = self.project_s(self.s - self.b2 * self.gs, _S)  # update formula for reorder level
            self.S = _S

    def forward(self):
        '''function for conducting forward step for updating agent internal state'''
        self.t += 1  # increment timing

        self.record_s.append(self.s)
        self.record_S.append(self.S)
        # store updated policy parameters

        if self.stationary != "stationary":
            rate = self.alpha / self.prev_alpha
            if rate > threshold_multiplier or rate < 1 / threshold_multiplier:
                # adaptation timing
                self.t = init_t

        self.a = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['a_rate']  # adaptation rate for relative value

        self.eta =  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_nom']) + 1) \
                    ** adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_mul']
        # adaptation rate for value function

        self.b1 = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b1_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b1_nom']) + 1) \
                    ** adap_rate[str(self.N)+ self.alg_type + self.stationary + self.demand_dist]['b1_mul']
        # adaptation rate for order-up-to level

        self.b2 = adap_rate[str(self.N)+ self.alg_type + self.stationary + self.demand_dist]['b2_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_nom']) + 1) \
                    ** adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_mul']
        # adaptation rate for reorder level

        if self.stationary == "stationary":  # for non-stationary scenario
            self.tau = self.init_tau / (np.int32(self.t / 10) + 1) ** 0.8
            self.sigma = self.init_sigma / (np.int32(self.t / 10) + 1) ** 0.8
            # adaptively decreasing of stochastic factor
        else:
            self.prev_alpha = self.alpha
            self.prev_beta = self.beta


    def copy_agent(self, void_agent):
        '''clone validation agent from original agent for validation process'''
        void_agent.N = self.N
        void_agent.s = self.s.copy()
        void_agent.S = self.S.copy()
        void_agent.alg_type = self.alg_type
        void_agent.demand_dist = self.demand_dist
        void_agent.prev_alpha = void_agent.alpha = self.alpha
        void_agent.prev_beta = void_agent.beta = self.beta
        void_agent.init_tau = void_agent.tau = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['tau']
        void_agent.init_sigma = void_agent.sigma = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['sigma']
        void_agent.mt = adap_rate[str(self.N) + self.alg_type + "non-stationary"  + self.demand_dist]['mt']

        void_agent.rho = self.rho
        void_agent.cnt = self.cnt.copy()
        void_agent.w = self.w.copy()