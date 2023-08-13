import numpy as np
from collections import deque
from Agent.util.hyper_param_set import *
# import library
np.random.seed(0)

class Agent:
    """
    Structured RL for Two-item inventory system

    :param N: (int) # of items
    :param s: (float) initial reorder level
    :param c: (float) initial can-order level
    :param S: (float) initial order-up-to level
    :param alpha: (float) initial estimated distributional parameter (first parameter)
    :param beta: (float) initial estimated distributional parameter (second parameter)
    :param alg_type: (string) assign type whether partial SA (SRL-PSA) or full SA (SRL-FSA)
    :param stationary: (string) assign mode whether stationary system (stationary) or not (non-stationary)
    :param demand_dist: (string) assign type of demand distribution (gamma, normal, poisson)
    """
    def __init__(self,N, s, c, S, alpha, beta, alg_type = "SRL-PSA", stationary = "stationary", demand_dist = "gamma"):

        self.N = N   # assign number of items
        self.s = np.ones(self.N) * s   # initialize reorder level
        self.c = np.ones(self.N) * c   # initialize can-order level
        self.S = np.ones(self.N) * S   # initialize order-up-to level
        self._s = self.s.copy()
        self._c = self.c.copy()
        self._S = self.S.copy()

        self.demand_dist = demand_dist   # assign distributional type mode
        self.h = h    # normalize factor for input on the relative value function
        self.cnt = deque(maxlen = max_len)  # define the observation queue

        self.alg_type = alg_type  # assign type of algorithm
        self.prev_alpha = self.alpha = alpha # estimated first parameter for underlying distribution
        self.prev_beta = self.beta = beta   # estimated second parameter for underlying distribution
        self.stationary = stationary   # assign mode whether stationary system or not

        self.dim = 14   # dimension for relative value function (2 dim 4th order polynomial regression => 14)

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

        self.record_s = [self.s.tolist()]
        self.record_c = [self.c.tolist()]
        self.record_S = [self.S.tolist()]
        # record queue for tracking the trajectories for each policy parameter

        self.multi_index = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
        # power index combination for two-dimensional value function

        self.gS = np.zeros(self.N)
        self.gc = np.zeros(self.N)
        self.gs = np.zeros(self.N)  #  initialize momentum variable

        self.w = np.zeros(self.dim)    # initialize parameter for relative value function
        self.w[-1] = low_pos   # initialize the parameter for highest order of each input with small positive value
        self.w[-5] = low_pos

        self.t = 0   # initialize time index
        self.rho = 0.   # initialize relative value
        self.forward()   # conduct initial forward step

    def init(self):
        '''function for initializing additional setting'''
        if self.alg_type == 'SRL-FSA':
            self.ms = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['ms']
            self.mc = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['mc']
            self.mS = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['mS']
            #  size of batch sample for policy update using full SA (FSA)

    def get_action(self, x, stochastic = True):
        '''function for producing action by following the policy'''
        noise = np.ones(self.N) * np.nan    # initialize noise term
        a = np.zeros(self.N)
        if stochastic:   # if stochastic (s,c S) replenishment mode
            f11 = self.f(x[0], self.s[0])
            f12 = self.f(x[0], self.c[0])
            f21 = self.f(x[1], self.s[1])
            f22 = self.f(x[1], self.c[1])   # assign sigmoid mixing probability on current state x with given policy parameter

            u11, u12, u21, u22 = np.random.rand(4)   # sampling from uniform
            noise_S1 = np.random.normal(loc = self.S[0], scale = self.sigma)
            noise_S2 = np.random.normal(loc = self.S[1], scale = self.sigma)    # sampling from normal (S_n, sigma)
            noise[0] = noise_S1 - self.S[0]
            noise[1] = noise_S2 - self.S[1]

            if (u11 > f21 and u12 > f12) or (u11 <= f21 and u12 > f11):  # P_1
                a[0] = np.maximum(noise_S1 - x[0], 0.)   # if negative value then zero clipping

            if (u21 > f11 and u22 > f22) or (u21 <= f11 and u22 > f21):  # P_1
                a[1] = np.maximum(noise_S2 - x[1], 0.)    # if negative value then zero clipping
        else:     # if deterministic (s,c S) replenishment mode
            if x[0] < self.s[0] or x[1] < self.s[1]:
                if x[0] < self.c[0]:
                    a[0] = self.S[0] - x[0]

                if x[1] < self.c[1]:
                    a[1] = self.S[1] - x[1]

        return a, noise

    def poly_basis(self, x1, x2):
        '''function for computing polynomial basis'''
        x1 /= self.h
        x2 /= self.h  # normalize inventory level input

        return np.array([x1, x2, x1 ** 2, x1 * x2, x2 ** 2,  x1 ** 3, x1**2 * x2, x1 * x2**2, x2 ** 3, x1 ** 4, x1** 3 * x2, x1 ** 2 * x2 ** 2, x1 * x2 ** 3, x2 ** 4], dtype=np.float32)


    def V(self, x1, x2):
        '''function for computing relative value'''
        pi = self.poly_basis(x1, x2)
        valuef = np.dot(self.w, pi)

        return valuef


    def update_V(self, x, r, _x):
        '''function for updating relative value function'''
        TD_target = r + self.V(*_x) - self.rho
        self.rho += self.a * (r + self.V(*_x) - self.V(*x) - self.rho)

        self.w += self.eta * (TD_target - self.V(*x)) * self.poly_basis(*x)
        self.w[-1] = np.maximum(0., self.w[-1])
        self.w[-5] = np.maximum(0., self.w[-5])    # restrict to highest order term as nonnegative value

    def f(self, x, y):
        '''sigmoid function in point x given s and tau value'''
        return 1 / (1+np.exp(-(x-y) / self.tau))

    def dfds(self, x, y):
        '''function for computing the derivative value of the sigmoid function with respect to reorder level'''
        sig = self.f(x, y)

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
        order = idx

        if order == 0:
            res = 1.
        elif order ==1:
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
        self.cnt.extend(obs.tolist())
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

    def get_P1(self, size, alpha, S, eps):
        '''function for generating random sample from the distribution P1
        which is the transition probability under inventory being below reorder level '''
        sample_x = S + eps - np.random.gamma(alpha, 1 / self.beta, size)

        return sample_x

    def get_PbarS(self, ber, S, eps, size):
        '''function for generating random sample from the composite distribution two different parameterized P1
        which is distinguished by random variable from the bernoulli, represented as follow (1- B) * P1(x;a-1) + B * P1(x;a)'''
        sample_x = (1 - ber) * self.get_P1(size, self.alpha - 1, S, eps) + ber * self.get_P1(size, self.alpha, S, eps)

        return sample_x

    def get_Pbars(self, x, ber, S, eps, size):
        '''function for generating random sample from the composite distribution two different parameterized P1
        which is distinguished by random variable from the bernoulli, represented as follow (1- B) * P0(x;a) + B * P1(x;a)'''
        sample_x = ber * self.get_P1(size, self.alpha, S, eps) + (1 - ber) * self.get_P0(x, size)

        return sample_x

    def project_s(self, x, c):
        '''projection operator to restrict the following equation s =< c'''
        return np.minimum(x, c)

    def project_c(self, x, S):
        '''projection operator to restrict the following equation c =< S'''
        return np.minimum(x, S)

    def update_policy(self, x, _x, eps):
        '''function for updating the policy parameters'''
        if self.alg_type == "SRL-PSA":   # if partial stochastic approximation (PSA) based algorithm

            A1_1 = (1 - self.f(x[0], self.s[0])) * self.f(x[1], self.c[1]) + self.f(x[0], self.s[0]) * self.f(x[1], self.s[1])
            A1_2 = (1 - self.f(x[0], self.s[0])) * (1 - self.f(x[1], self.c[1])) + self.f(x[0], self.s[0]) * (1 - self.f(x[1], self.s[1]))

            A2_1 = (1 - self.f(x[1], self.s[1])) * self.f(x[0], self.c[0]) + self.f(x[1], self.s[1]) * self.f(x[0], self.s[0])
            A2_2 = (1 - self.f(x[1], self.s[1])) * (1 - self.f(x[0], self.c[0])) + self.f(x[1], self.s[1]) * (1 - self.f(x[0], self.s[0]))
            # fixed term updating for S

            c1_chg = np.sum([self.w[j + np.int32((i + j) * (i + j + 1)) // 2 - 1] * (self.get_exp_diff(idx=i, alpha=self.alpha, x = x[0]) - self.get_exp_diff(idx=i, alpha=self.alpha, x=self.S[0] + eps[0]))
                             * (A1_1 * self.get_exp_diff(idx=j, alpha=self.alpha, x = x[1]) + A1_2 * self.get_exp_diff(idx=j,alpha=self.alpha, x=self.S[1] + eps[1])) for i, j in self.multi_index[1:]])

            c2_chg = np.sum([self.w[j + np.int32((i + j) * (i + j + 1)) // 2 - 1] * (self.get_exp_diff(idx = j, alpha=self.alpha, x = x[1]) - self.get_exp_diff(idx=j, alpha=self.alpha, x=self.S[1] + eps[1]))
                             * (A2_1 * self.get_exp_diff(idx=i, alpha=self.alpha, x=x[0]) + A2_2 * self.get_exp_diff(idx = i, alpha=self.alpha, x=self.S[0] + eps[0])) for i, j in self.multi_index[1:]])
            # shared term updating for c and s

            self.gS[0] = self.mt * self.gS[0] + (1 - self.mt) * self.beta * ((1 - self.f(x[1], self.s[1])) * (1 - self.f(x[0], self.c[0])) + self.f(x[1], self.s[1]) * (1 - self.f(x[0], self.s[0]))) \
                  * np.sum([self.w[j + np.int32((i + j) * (i + j + 1)) // 2 - 1] * (self.get_exp_diff(idx = i, alpha = self.alpha - 1, x = self.S[0] + eps[0]) - self.get_exp_diff(idx = i, alpha = self.alpha, x = self.S[0] + eps[0]))
                            * (A1_1 * self.get_exp_diff(idx = j, alpha = self.alpha, x = x[1]) + A1_2 * self.get_exp_diff(idx = j, alpha = self.alpha, x = self.S[1] + eps[1])) for i, j in self.multi_index[1:]])


            self.gS[1] = self.mt * self.gS[1] + (1 - self.mt) * self.beta * ((1 - self.f(x[0], self.s[0])) * (1 - self.f(x[1], self.c[1])) + self.f(x[0], self.s[0]) * (1 - self.f(x[1], self.s[1]))) \
                  * np.sum([self.w[j + np.int32((i + j) * (i + j + 1)) // 2 - 1] * (self.get_exp_diff(idx = j, alpha = self.alpha - 1, x = self.S[1] + eps[1]) - self.get_exp_diff(idx = j, alpha = self.alpha, x = self.S[1] + eps[1]))
                            * (A2_1 * self.get_exp_diff(idx = i, alpha = self.alpha, x = x[0]) + A2_2 * self.get_exp_diff(idx = i, alpha = self.alpha, x = self.S[0] + eps[0])) for i, j in self.multi_index[1:]])
            # momentum for updating S

            self._S = self.S - self.b1 * np.clip(self.gS, -bigM, bigM)  # update formula for order-up-to level

            self.gc[0] = self.mt * self.gc[0] + (1 - self.mt) * (1 - self.f(x[1], self.s[1])) * self.dfds(x[0], self.c[0]) * (c1_chg)
            self.gc[1] = self.mt * self.gc[1] + (1 - self.mt) * (1 - self.f(x[0], self.s[0])) * self.dfds(x[1], self.c[1]) * (c2_chg)
            # momentum for updating c

            self._c = self.project_c(self.c - self.b2 * np.clip(self.gc , -bigM, bigM), self._S)   # update formula for can-order level

            self.gs[0] = self.mt * self.gs[0] + (1 - self.mt) * self.dfds(x[0], self.s[0]) * (self.f(x[1], self.s[1]) * c1_chg + (self.f(x[1], self.s[1]) - self.f(x[1], self.c[1])) * c2_chg)
            self.gs[1] = self.mt * self.gs[1] + (1 - self.mt) * self.dfds(x[1], self.s[1]) * (self.f(x[0], self.s[0]) * c2_chg + (self.f(x[0], self.s[0]) - self.f(x[0], self.c[0])) * c1_chg)
            # momentum for updating s

            self._s = self.project_s(self.s - self.b3 * np.clip(self.gs , -bigM, bigM), self._c)    # update formula for reorder level

            self.S = self._S.copy()
            self.c = self._c.copy()
            self.s = self._s.copy()

        else:    # if full stochastic approximation (FSA) based algorithm
            cls_S = np.random.binomial(size = 2 * self.mS, p = 0.5, n = 1)  # sampling from underlying distribution for gradient approximation of order-up-to level

            yS1_1 = cls_S[:self.mS] * self.get_P1(self.mS, self.alpha, self.S[0], eps[0]) + (1 - cls_S[:self.mS]) * self.get_P1(self.mS, self.alpha - 1, self.S[0], eps[0])
            # sampling from composite distribution
            yS1_2 = _x[1] * np.ones_like(yS1_1) # assign next observed state

            self.gS[0] = self.mt * self.gS[0] + (1 - self.mt) * ((1 - self.f(x[1], self.s[1])) * (1 - self.f(x[0], self.c[0])) + self.f(x[1], self.s[1]) * (1 - self.f(x[0], self.s[0]))) * np.mean((-1) ** cls_S[:self.mS] * self.V(yS1_1, yS1_2))

            yS2_1 = _x[0] * np.ones_like(yS1_1) # assign next observed state
            yS2_2 = cls_S[self.mS:] * self.get_P1(self.mS, self.alpha, self.S[1], eps[1]) + (1 - cls_S[self.mS:]) * self.get_P1(self.mS, self.alpha - 1, self.S[1], eps[1])
            # sampling from composite distribution

            self.gS[1] = self.mt * self.gS[1] + (1 - self.mt) * ((1 - self.f(x[0], self.s[0])) * (1 - self.f(x[1], self.c[1])) + self.f(x[0], self.s[0]) * (1 - self.f(x[1], self.s[1]))) * np.mean((-1) ** cls_S[self.mS:] * self.V(yS2_1, yS2_2))
            self._S = self.S - self.b1 * self.gS    # update formula for order-up-to level


            cls_c = np.random.binomial(size = 2 * self.mc, p = 0.5, n = 1)    # sampling from underlying distribution for gradient approximation of can-order level

            yc1_1 = cls_c[:self.mc] * self.get_P1(self.mc, self.alpha, self.S[0], eps[0]) + (1 - cls_c[:self.mc]) * self.get_P0(x[0], self.mc)
            yc1_2 = _x[1] * np.ones_like(yc1_1)

            self.gc[0] = self.mt * self.gc[0] + (1 - self.mt) * (1 - self.f(x[1], self.s[1])) * self.dfds(x[0], self.c[0]) * np.mean((-1) ** cls_c[:self.mc] * self.V(yc1_1, yc1_2))

            yc2_2 = cls_c[self.mc:] * self.get_P1(self.mc, self.alpha, self.S[1], eps[1]) + (1 - cls_c[self.mc:]) * self.get_P0(x[1], self.mc)
            yc2_1 = _x[0] * np.ones_like(yc2_2)

            self.gc[1] = self.mt * self.gc[1] + (1 - self.mt) * (1 - self.f(x[0], self.s[0])) * self.dfds(x[1], self.c[1]) * np.mean((-1) ** cls_c[self.mc:] * self.V(yc2_1, yc2_2))
            self._c = self.project_c(self.c - self.b2 * self.gc, self._S)  # update formula for can-order level

            cls_s = np.random.binomial(size = 2 * self.ms, p = 0.5, n=1)  # sampling from Bernoulli with equal prob
            z11 = self.get_P1(self.ms, self.alpha, self.S[0], eps[0])
            z12 = self.get_P0(x[0], self.ms)
            z1 = np.where(cls_s[:self.ms], z11, z12)  # sampling from composite distribution

            z21 = self.get_P1(self.ms, self.alpha, self.S[1], eps[1])
            z22 = self.get_P0(x[1], self.ms)
            z2 = np.where(cls_s[self.ms:], z21, z22)   # sampling from composite distribution

            theta1 = self.f(x[1], self.s[1]) / (2 * self.f(x[1], self.s[1]) - self.f(x[1], self.c[1]))
            # parameter for Bernoulli for item1

            e1 = np.random.binomial(size = self.ms, p=theta1, n = 1)   # sampling from Bernoulli

            ys1_1 = np.where(e1, z1, _x[0])
            ys1_2 = np.where(e1, _x[1], z2)

            self.gs[0] = self.mt * self.gs[0] + (1 - self.mt) * self.dfds(x[0], self.s[0]) * np.mean((-1) ** (e1 * cls_s[:self.ms] + (1 - e1) * cls_s[self.ms:]) * self.V(ys1_1, ys1_2))

            theta2 = self.f(x[0], self.s[0]) / (2 * self.f(x[0], self.s[0]) - self.f(x[0], self.c[0]))
            # parameter for Bernoulli for item2

            e2 = np.random.binomial(size = self.ms, p = theta2, n = 1)   # sampling from Bernoulli

            ys2_1 = np.where(e2, _x[0], z1)
            ys2_2 = np.where(e2, z2, _x[1])

            self.gs[1] = self.mt * self.gs[1] + (1 - self.mt) * self.dfds(x[1], self.s[1]) * np.mean((-1) ** (e2 * cls_s[self.ms:] + (1 - e2) * cls_s[:self.ms]) * self.V(ys2_1, ys2_2))
            self._s = self.project_s(self.s - self.b3 * np.clip(self.gs, -bigM, bigM), self._c)  # update formula for reorder level
            self.S = self._S.copy()
            self.c = self._c.copy()
            self.s = self._s.copy()

    def forward(self):
        '''function for conducting forward step for updating agent internal state'''
        self.t += 1   # increment timing

        self.record_s.append(self.s.tolist())
        self.record_c.append(self.c.tolist())
        self.record_S.append(self.S.tolist())
        # store updated policy parameters

        if self.stationary != "stationary":
            rate = self.alpha / self.prev_alpha
            if rate > threshold_multiplier or rate < 1 / threshold_multiplier:
                # adaptation timing
                self.t = init_t

        self.a = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['a_rate']   # adaptation rate for relative value

        self.eta =  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_nom']) + 1) \
                    **  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['eta_mul']
        # adaptation rate for value function

        self.b1 = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b1_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b1_nom']) + 1) \
                    **  adap_rate[str(self.N)+ self.alg_type + self.stationary + self.demand_dist]['b1_mul']
        # adaptation rate for order-up-to level

        self.b2 = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_nom']) + 1) \
                    **  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b2_mul']
        # adaptation rate for can-order level

        self.b3 = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b3_denom'] / \
                    (np.int32(self.t /  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b3_nom']) + 1) \
                    **  adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['b3_mul']
        # adaptation rate for reorder level

        if self.stationary == "stationary":   # for non-stationary scenario
            self.tau = self.init_tau / (np.int32(self.t / 10) + 1) ** 0.8
            self.sigma = self.init_sigma / (np.int32(self.t / 10) + 1) ** 0.8
            # adaptively decreasing of stochastic factor
        else:
            if self.alg_type == "SRL-PSA":
                self.tau = self.init_tau / (np.int32((self.t) / 10) + 1)
                self.sigma = self.init_sigma / (np.int32((self.t) / 10) + 1)
            self.prev_alpha = self.alpha
            self.prev_beta = self.beta



    def copy_agent(self, void_agent):
        '''clone validation agent from original agent for validation process (non-stationary)'''
        void_agent.s = self.s.copy()
        void_agent.c = self.c.copy()
        void_agent.S = self.S.copy()
        void_agent.alg_type = self.alg_type
        void_agent.demand_dist = self.demand_dist
        void_agent.prev_alpha = void_agent.alpha = self.alpha
        void_agent.prev_beta = void_agent.beta = self.beta
        void_agent.init_tau = void_agent.tau = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['tau']
        void_agent.init_sigma = void_agent.sigma = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['sigma']
        void_agent.mt = adap_rate[str(self.N) + self.alg_type + "non-stationary" + self.demand_dist]['mt']

        void_agent.rho = self.rho
        void_agent.cnt = self.cnt.copy()
        void_agent.w = self.w.copy()

