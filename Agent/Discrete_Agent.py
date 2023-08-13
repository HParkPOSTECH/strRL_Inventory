import numpy as np
import scipy.stats as stats
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
    :param beta: (int) maximum demand for truncated demand distribution
    :param alg_type: (string) assign type whether partial SA (SRL-PSA) or full SA (SRL-FSA)
    :param stationary: (string) assign mode whether stationary system (stationary) or not (non-stationary)
    :param demand_dist: (string) assign type of demand distribution (gamma, normal, poisson)
    """
    def __init__(self, N, s, S, beta, capa, alg_type = "SRL-FSA", stationary = "stationary", demand_dist = "gamma"):
        self.N = N  # assign number of items
        self.s = s * np.ones(N)  # initialize reorder level
        self.S = S * np.ones(N)  # initialize order-up-to level
        self._s = self.s.copy()
        self._S = self.S.copy()

        self.stationary = stationary  # assign mode whether stationary system or not
        self.demand_dist = demand_dist   # assign distributional type mode
        self.alg_type = alg_type  # assign type of algorithm
        self.capa = capa # maximum capacity
        self.max_d = beta # maximum demand for truncated demand distribution
        self.h = capa // 2   # normalize factor for input on the relative value function
        self.obs_queue = deque(maxlen = max_len)   # define the observation queue
        self.cnt = np.zeros(self.max_d + 1)  # observed demand count variable
        self.prob = np.ones(self.max_d + 1) / (self.max_d + 1) # proportion of observed demand
        self.dim = 5  # dimension for relative value function (=5 for 4th order polynomial regression with intersection term)

        self.init_tau = self.tau = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['tau']
        # initialize tau as a hyper-parameter of sigmoid function
        self.init_sigma = self.sigma = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['sigma']
        # initialize sigma for standard deviation of noise distribution in stochastic policy
        self.mt = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['mt']  # momentum update factor

        self.gS = self.gs = 0.  # initialize momentum variable

        self.w = np.zeros(self.dim)   # initialize parameter for relative value function
        self.w[-1] = low_pos   # initialize the parameter for highest order term with small positive value

        self.t = 0  # initialize time index
        self.diff = 0.5  # minimum difference between S <-> s
        self.rho = 0.  # initialize relative value

        self.record_s = [self.s]
        self.record_S = [self.S]
        # record queue for tracking the trajectories for each policy parameter

        self.forward()  # conduct initial forward step

    def init(self):
        '''function for initializing additional setting'''
        if self.alg_type == 'SRL-FSA':
            self.m = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['m']
            #  size of batch sample for policy update using full SA (FSA)


    def get_action(self, x, stochastic = True):
        '''function for producing action by following the policy'''
        noise = None  # initialize noise term

        if stochastic:  # if stochastic (s,S) replenishment mode
            p = self.f(x)   # assign sigmoid mixing probability on current state x
            u = np.random.rand(1)  # sampling from uniform
            noise_S = stats.truncnorm(self.s, self.capa, loc = self.S, scale = self.sigma).rvs(1)[0]  #  sampling from truncated normal (S, sigma | capa)
            noise = noise_S - self.S
            if p < u: # P_1
                a = np.maximum(np.round(noise_S - x), 0)  # if negative value then zero clipping
            else: # P_0
                a = 0

        else:  # if deterministic (s,S) replenishment mode
            if x < self.s:
                a = np.round(self.S - x)
            else:
                a = 0

        return a, noise

    def poly_basis(self, x):
        '''function for computing polynomial basis'''
        return np.array([(x / self.h) ** i  for i in range(self.dim)], dtype=np.float32)

    def integral_poly_basis(self, x, multiplied_order = 0):
        '''integrated polynomial basis'''
        return np.array([(x / self.h) ** (multiplied_order + i + 1) / (multiplied_order + i + 1) for i in range(self.dim)], dtype=np.float32)

    def V(self, x):
        '''function for computing relative value'''
        pi = self.poly_basis(x)
        valuef = np.dot(self.w, pi)

        return valuef

    def integral_V(self, x, multiplied_order = 0):
        '''function for integrated relative value function'''
        pi = self.integral_poly_basis(x, multiplied_order=multiplied_order)

        return np.dot(self.w, pi)

    def integral_dP1dS_V(self, eps):
        '''integration of function for derivate of P1 w.r.t S multiplied with Value function'''
        res = 0.
        dP1dS = np.diff(self.norm_prob) # derivative of P1 w.r.t S parameter
        pre, cur = None, None

        for i in range(self.max_d):

            if i == 0:
                cur = self.integral_V(self.S + eps)
                pre = self.integral_V((self.S + eps - 1))
            else:
                cur = pre
                pre = self.integral_V((self.S + eps - i - 1))

            res += dP1dS[i] * (cur - pre)

        return res

    def integral_P0_V(self, x):
        '''integration Value function under P0 probability for given point x - Y'''
        res = 0.
        norm_diff = np.diff(self.norm_prob)  # differentiated demand proportion
        pre, cur = None, None
        m_pre, m_cur = None, None

        for i in range(self.max_d):
            if i == 0:
                cur = self.integral_V(x)
                pre = self.integral_V((x - 1))

                m_cur = self.integral_V(x, multiplied_order=1)
                m_pre = self.integral_V((x - 1), multiplied_order=1)

            else:
                cur = pre
                pre = self.integral_V((x - i - 1))

                m_cur = m_pre
                m_pre = self.integral_V((x - i - 1), multiplied_order=1)

            res += - norm_diff[i] * (m_cur - m_pre) + (norm_diff[i] * (x - i) + self.norm_prob[i]) * (cur - pre)

        return res

    def integral_P1_V(self, eps):
        '''integration Value function under P1 probability for given point S + eps - Y'''
        res = 0.
        norm_diff = np.diff(self.norm_prob)  # differentiated demand proportion
        pre, cur = None, None
        m_pre, m_cur = None, None

        for i in range(self.max_d):
            if i == 0:
                cur = self.integral_V(self.S + eps)
                pre = self.integral_V((self.S + eps - 1))

                m_cur = self.integral_V(self.S + eps, multiplied_order=1)
                m_pre = self.integral_V((self.S + eps - 1), multiplied_order=1)
            else:
                cur = pre
                pre = self.integral_V((self.S + eps - i - 1))

                m_cur = m_pre
                m_pre = self.integral_V((self.S + eps - i - 1), multiplied_order=1)

            res += - norm_diff[i] * (m_cur - m_pre) + (norm_diff[i] * (self.S + eps - i) + self.norm_prob[i]) * (cur - pre)

        return res


    def update_V(self, x, r, _x):
        '''function for updating relative value function'''
        TD_target = r + self.V(_x[0]) - self.rho

        self.rho += self.alpha * (r + self.V(_x[0]) - self.V(x[0]) - self.rho)
        self.w += self.eta * ((TD_target - self.V(x[0])) * self.poly_basis(x[0]))
        self.w[-1] = np.maximum(1e-6, self.w[-1])  # restrict to highest order term as nonnegative value

    def f(self, x):
        '''sigmoid function in point x given s and tau value'''
        return 1. / (1 + np.exp(-(x-self.s) / self.tau))

    def dfds(self, x):
        '''function for computing the derivative value of the sigmoid function with respect to reorder level'''
        sig = self.f(x)
        return sig * (sig - 1) / self.tau

    def step(self, x):
        '''step function'''
        return np.where(x > 0, np.zeros_like(x), np.ones_like(x))

    def adap_param(self, obs, warmup):
        '''function for adapting the estimation of the underlying parameters of gamma distribution'''
        self.obs_queue.append(obs)

        self.cnt[np.int32(obs)] += 1
        self.prob = self.cnt / np.sum(self.cnt)

        norm_const = np.sum(self.prob[1:-1]) + (self.prob[0] + self.prob[-1]) / 2.
        self.norm_prob = self.prob / norm_const

    def integral_piecewise_linear(self):
        '''get cumulative probability for discrete supports'''
        boundary = np.zeros_like(self.norm_prob)
        dp = np.diff(self.norm_prob)
        for i in range(self.norm_prob.shape[0] - 1):
            boundary[i + 1] = dp[i] / 2. * (i + 1) ** 2 + (self.norm_prob[i] - dp[i] * i) * (i + 1)  - (dp[i] / 2. * (i) ** 2 + (self.norm_prob[i] - dp[i] * i) * (i))

        return boundary.cumsum()

    def inverse_sample(self, size):
        '''inverse cumulative sampling for piecewise approximated demand distribution'''
        bdx = self.integral_piecewise_linear()
        dp = np.diff(self.norm_prob)
        U = np.random.rand(size)  # uniform sample
        sample = np.zeros_like(U)
        for idx, u in enumerate(U):
            for i in range(self.norm_prob.shape[0] - 1):
                if bdx[i] <= u < bdx[i + 1]: # if uniform sample is laid in the discrete cdf range ==> select
                    a = np.where(dp[i] == 0, dp[i] + 1e-8, dp[i]) / 2.
                    b = self.norm_prob[i] - dp[i] * i
                    c = bdx[i] - (a * i ** 2 + b * i)
                    c -= u
                    sample[idx] = (-b + (b ** 2 - 4 * a * c) ** .5) / a / 2

        return sample

    def get_P0(self, x):
        '''function for generating random sample from the distribution P0
        which is the transition probability under inventory exceeding reorder level '''
        sample_x = x - self.inverse_sample(1)

        return sample_x

    def get_P1(self, eps):
        '''function for generating random sample from the distribution P1
        which is the transition probability under inventory being below reorder level '''
        sample_x = self.S + eps  - self.inverse_sample(1)

        return sample_x

    def get_Pbar(self, x, eps, alpha):
        '''function for generating random sample from the composite distribution two different parameterized P1
        which is distinguished by random variable from the bernoulli, represented as follow (1- B) * P1(x;a-1) + B * P1(x;a)'''
        sample_x = alpha * self.get_P1(eps) + (1 - alpha) * self.get_P0(x)

        return sample_x

    def project_S(self, x):
        '''projection operator to restrict the following equation S =< capa'''
        return np.clip(x, -self.max_d + self.diff, self.capa)

    def project_s(self, x, S):
        '''projection operator to restrict the following equation s =< S'''
        return np.clip(x, -self.max_d + self.diff, S - self.diff)

    def update_policy(self, x, _x, eps):
        '''function for updating the policy parameters'''
        if self.alg_type == "SRL-PSA":    # if partial stochastic approximation (PSA) based algorithm (see Appendix M)

            self.gS = self.mt * self.gS + (1 - self.mt) * np.clip((1 - self.f(x)) * self.integral_dP1dS_V(eps), -bigM, bigM)
            self._S = self.project_S(self.S - self.beta1 * self.gS)  # update formula for order-up-to level

            self.gs = self.mt * self.gs + (1 - self.mt) * np.clip(self.dfds(x) * (self.integral_P0_V(x) - self.integral_P1_V(eps)), -bigM, bigM)
            self.s = self.project_s(self.s - self.beta2 * self.gs, self._S)  # update formula for reorder level
            self.S = self._S
        else:    # if full stochastic approximation (FSA) based algorithm  (see Appendix L)
            n = np.random.choice(self.max_d, p=np.abs(np.diff(self.norm_prob)) / np.sum(np.abs(np.diff(self.norm_prob))), size=self.m)
            alpha = np.random.binomial(size=self.m, p=.5, n=1)  # sampling from underlying distribution for gradient approximation of order-up-to level

            z = np.zeros(self.m)
            y = np.zeros(self.m)

            for i in range(self.m):
                z[i] = np.random.uniform(self.S + eps - n[i] - 1, self.S + eps - n[i])
                y[i] = self.get_Pbar(x, eps, alpha[i])

            self.gS = self.mt * self.gS + (1 - self.mt) * np.mean((1 - self.f(x)) * (-1) ** self.step(np.tile(np.diff(self.norm_prob)[np.newaxis], (self.m, 1))[:, n]) * self.V(z))
            _S = self.project_S(self.S - self.beta1 * self.gS)   # update formula for order-up-to level

            self.gs = self.mt * self.gs + (1 - self.mt) * np.mean(self.dfds(x) * (-1) ** alpha * self.V(y))
            self.s = self.project_s(self.s - self.beta2 * self.gs, _S)  # update formula for reorder level

            self.S = _S


    def forward(self):
        '''function for conducting forward step for updating agent internal state'''
        self.t += 1  # increment timing

        self.record_s.append(self.s)
        self.record_S.append(self.S)
        # store updated policy parameters

        self.alpha = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['a_rate']  # adaptation rate for relative value

        self.eta =  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_nom']) + 1) \
                    ** adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_mul']
        # adaptation rate for value function

        self.beta1 = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b1_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b1_nom']) + 1) \
                    ** adap_rate[str(self.N)+ self.alg_type + self.stationary + self.demand_dist]['b1_mul']
        # adaptation rate for order-up-to level

        self.beta2 = adap_rate[str(self.N)+ self.alg_type + self.stationary + self.demand_dist]['b2_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_nom']) + 1) \
                    ** adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_mul']
        # adaptation rate for reorder level

        self.tau = self.init_tau / (np.int32(self.t / 10) + 1) ** 0.8
        self.sigma = self.init_sigma / (np.int32(self.t / 10) + 1) ** 0.8
        # adaptively decreasing of stochastic factor


    def copy_agent(self, void_agent):
        '''clone validation agent from original agent for validation process'''
        void_agent.s = self.s
        void_agent.S = self.S
        void_agent.rho = self.rho
        void_agent.alg_type = self.alg_type
        void_agent.demand_dist = self.demand_dist
        void_agent.prev_beta = void_agent.beta = self.max_d
        void_agent.init_tau = void_agent.tau = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['tau']
        void_agent.init_sigma = void_agent.sigma = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['sigma']
        void_agent.mt = adap_rate[str(self.N) + self.alg_type + "non-stationary"  + self.demand_dist]['mt']

        void_agent.rho = self.rho
        void_agent.cnt = self.cnt.copy()
        void_agent.w = self.w.copy()


