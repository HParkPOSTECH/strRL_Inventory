import numpy as np
from collections import deque
from itertools import product
from Agent.util.hyper_param_set import *
# import library
np.random.seed(0)

class Agent:
    """
    Structured RL for Four-item inventory system

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
    def __init__(self, N, s, c, S,  alpha, beta, alg_type = "SRL-FSA", stationary = "stationary", demand_dist = "gamma"):
        self.N = N   # assign number of items
        self.s = np.ones(self.N) * s  # initialize reorder level
        self.c = np.ones(self.N) * c   # initialize can-order level
        self.S = np.ones(self.N) * S  # initialize order-up-to level
        self._s = self.s.copy()
        self._c = self.c.copy()
        self._S = self.S.copy()

        self.alg_type = alg_type  # assign type of algorithm
        self.prev_alpha = self.alpha = alpha  # estimated first parameter for underlying distribution
        self.prev_beta = self.beta = beta  # estimated second parameter for underlying distribution
        self.stationary = stationary  # assign mode whether stationary system or not
        self.demand_dist = demand_dist  # assign distributional type mode

        self.h = h   # normalize factor for input on the relative value function
        self.cnt = deque(maxlen = max_len)  # define the observation queue

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


        self.max_degree = value_degree + 1  #  number of term in value function for each item
        self.w = np.zeros((self.max_degree, self.max_degree, self.max_degree, self.max_degree))  # initialize parameter for relative value function
        self.w[-1, 0, 0, 0] = low_pos   # initialize the parameter for highest order of each input with small positive value
        self.w[0, -1, 0, 0] = low_pos
        self.w[0, 0, -1, 0] = low_pos
        self.w[0, 0, 0, -1] = low_pos

        self.gS = np.zeros(self.N)
        self.gc = np.zeros(self.N)
        self.gs = np.zeros(self.N) #  initialize momentum variable

        self.t = 0   # initialize time index
        self.rho = 0.  # initialize relative value
        self.reg_w = 0.00001  # regulerized weight for value function

        self.forward()  # conduct initial forward step

    def init(self):
        '''function for initializing additional setting'''
        if self.alg_type == 'SRL-FSA':
            self.ms = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['ms']
            self.mc = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['mc']
            self.mS = adap_rate[str(self.N) + self.alg_type + self.stationary + self.demand_dist]['mS']
            #  size of batch sample for policy update using full SA (FSA)

    def get_action(self, x, stochastic = True):
        '''function for producing action by following the policy'''
        noise = np.ones(self.N) * np.nan  # initialize noise term
        a = np.zeros(self.N)

        if stochastic:   # if stochastic (s,c S) replenishment mode
            f1 = self.f(x, self.s)
            f2 = self.f(x, self.c)   # assign sigmoid mixing probability on current state x with given policy parameter

            u1 = np.random.rand(self.N)
            u2 = np.random.rand(self.N)    # sampling from uniform
            noise_S = np.random.multivariate_normal(self.S, np.identity(self.N) * self.sigma)    # sampling from normal (S_n, sigma)
            noise = noise_S - self.S

            if (u1 > f1).any():  # P_1
                for n in range(self.N):
                    if u2[n] > f2[n]:
                        a[n] = np.maximum(noise_S[n] - x[n], 0.)   # if negative value then zero clipping

        else:   # if deterministic (s,c S) replenishment mode
            if (x < self.s).any():
                for n in range(self.N):
                    if x[n] < self.c[n]:
                        a[n] = self.S[n] - x[n]


        return a, noise

    def poly_basis(self, x, batch = False):
        '''function for computing polynomial basis'''
        temp_xs = x.copy()
        temp_xs = temp_xs / self.h   # normalize inventory level input

        if batch:  # for batch evaluation
            basis = np.zeros((x.shape[1], self.max_degree, self.max_degree, self.max_degree, self.max_degree), dtype=np.float32)

            for perm in product(np.arange(self.max_degree), repeat=self.N):
                if np.sum(perm) > value_degree:
                    continue

                basis[:, perm[0], perm[1], perm[2], perm[3]] = np.array([temp_xs[n, :] ** perm[n] for n in range(self.N)]) .prod(axis = 0)
                # compute term (x_1 ^i * x_2 ^ j * x_3 ^k  * x_4 ^l)

        else:
            basis = np.zeros((self.max_degree, self.max_degree, self.max_degree, self.max_degree), dtype=np.float32)

            for perm in product(np.arange(self.max_degree), repeat=self.N):
                if np.sum(perm) > value_degree:
                    continue

                basis[perm] = np.array([temp_xs[n] ** perm[n] for n in range(self.N)]) .prod()
                # compute term (x_1 ^i * x_2 ^ j * x_3 ^k  * x_4 ^l)

        return basis

    def V(self, x,  batch = False):
        '''function for computing relative value'''
        pi = self.poly_basis(x, batch=batch)
        if batch: # for batch mode
            valuef = np.sum(self.w[np.newaxis] * pi, axis = (1, 2, 3, 4))
        else:
            valuef = np.sum(self.w * pi)

        return valuef


    def update_V(self, x, r, _x):
        '''function for updating relative value function'''
        next_value = self.V(_x)
        TD_target = r + next_value - self.rho  # Temporal difference target
        self.rho += self.a * (r + next_value - self.V(x) - self.rho) # relative value update formula

        self.w += self.eta * (TD_target - self.V(x)) * self.poly_basis(x) - self.reg_w * self.w
        # value function update formula
        self.w[-1, 0, 0, 0] = np.maximum(0., self.w[-1, 0, 0, 0])
        self.w[0, -1, 0, 0] = np.maximum(0., self.w[0, -1, 0, 0])
        self.w[0, 0, -1, 0] = np.maximum(0., self.w[0, 0, -1, 0])
        self.w[0, 0, 0, -1] = np.maximum(0., self.w[0, 0, 0, -1])  # restrict to highest order term as nonnegative value

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
        self.cnt.extend(obs)
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

    def update_policy(self,  x, _x, eps):
        '''function for updating the policy parameters'''
        Sbase = np.zeros(self.N)
        cbase = np.zeros(self.N)
        sbase = np.zeros(self.N)

        for i in range(self.N):  # initialize term for updating each parameters
            Sbase[i] = (1 - self.f(x[i], self.c[i])) * (1 - np.prod([self.f(x[k], self.s[k]) for k in np.delete(np.arange(self.N), i)])) + (1 - self.f(x[i], self.s[i])) * np.prod([self.f(x[k], self.s[k]) for k in np.delete(np.arange(self.N), i)])
            cbase[i] = (1 - np.prod([self.f(x[k], self.s[k]) for k in np.delete(np.arange(self.N), i)])) * self.dfds(x[i], self.c[i])
            sbase[i] = self.dfds(x[i], self.s[i])

        if self.alg_type == "SRL-PSA":         # if partial stochastic approximation (PSA) based algorithm

            A1 = np.zeros(self.N)
            A2 = np.zeros(self.N)

            for i in range(self.N):
                sremain_prod = np.prod([self.f(x[j], self.s[j]) for j in np.delete(np.arange(self.N), i)])
                A1[i] = (1 - sremain_prod) * self.f(x[i], self.c[i]) + self.f(x[i], self.s[i]) * sremain_prod
                A2[i] = (1 - sremain_prod) * (1 - self.f(x[i], self.c[i])) + (1 - self.f(x[i], self.s[i])) * sremain_prod
            # fixed term updating for S

            dS = np.zeros(self.N)
            dc = np.zeros(self.N)
            for perm in product(np.arange(self.max_degree), repeat = self.N):  # compute for all permutation for (# of item X degree)
                if np.sum(perm) > self.N or np.sum(perm) == 0: # exclude for over degree or constant term
                    continue
                for n in range(self.N):
                    remain_prod = np.prod([A1[m] * self.get_exp_diff(idx=perm[m], alpha=self.alpha, x=x[m]) + A2[m] * self.get_exp_diff(idx=perm[m], alpha=self.alpha, x=self.S[m] + eps[m]) for m in np.delete(np.arange(self.N), n)])
                    dS[n] += self.w[perm] * (self.get_exp_diff(idx=perm[n], alpha=self.alpha - 1, x=self.S[n] + eps[n]) - self.get_exp_diff(idx=perm[n], alpha=self.alpha, x=self.S[n] + eps[n])) * remain_prod
                    dc[n] += self.w[perm] * (self.get_exp_diff(idx=perm[n], alpha=self.alpha, x=x[n]) - self.get_exp_diff(idx=perm[n], alpha=self.alpha, x=self.S[n] + eps[n])) * remain_prod
            # update term for c and S

            self.gS = self.mt * self.gS + (1 - self.mt) * self.beta * Sbase * dS  # momentum for updating S
            _S = self.S - self.b1 * self.gS   # update formula for order-up-to level

            self.gc = self.mt * self.gc + (1 - self.mt) * cbase * dc  # momentum for updating c
            _c = self.project_c(self.c - self.b2 * self.gc, _S)     # update formula for can-order level

            ds = np.zeros(self.N)
            for i in range(self.N):
                ds[i] = np.prod([self.f(x[m], self.s[m]) for m in np.delete(np.arange(self.N), i)]) * dc[i]
                for n in range(self.N):
                    if i == n:
                        continue
                    ds[i] += (self.f(x[n], self.s[n]) - self.f(x[n], self.c[n])) * dc[n] * np.prod([self.f(x[m], self.s[m]) for m in np.delete(np.arange(self.N), n)])
            # update term for s

            self.gs = self.mt * self.gs + (1 - self.mt) * sbase * ds   # momentum for updating s
            _s = self.project_s(self.s - self.b3 * self.gs, _c)   # update formula for reorder level

            self.S = _S.copy()
            self.c = _c.copy()
            self.s = _s.copy()

        else:     # if full stochastic approximation (FSA) based algorithm
            cls_S = np.random.binomial(size = self.N * self.mS, p = 0.5, n = 1)  # sampling from underlying distribution for gradient approximation of order-up-to level

            yS = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        yS[i, j] = cls_S[i] * self.get_P1(self.mS, self.alpha, self.S[i], eps[i]) + (1 - cls_S[i]) * self.get_P1(self.mS, self.alpha - 1, self.S[i], eps[i])
                    else:
                        yS[i, j] = _x[j]
            # sampling from composite distribution /  assign next observed state

            for n in range(self.N):
                self.gS[n] = self.mt * self.gS[n] + (1 - self.mt) * self.beta * Sbase[n] * np.mean((-1) ** cls_S[n] * self.V(yS[n]))  # momentum for updating S

            self._S = self.S - self.b1 * self.gS  # update formula for order-up-to level

            cls_c = np.random.binomial(size = (self.N, self.mc), p = 0.5, n = 1)

            yc = np.zeros((self.mc, self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        yc[:, i, j] = cls_c[i] * self.get_P1(self.mc, self.alpha, self.S[i], eps[i]) + (1 - cls_c[i]) * self.get_P0(x[i], self.mc)
                    else:
                        yc[:, i, j] = _x[j] * np.ones(self.mc)
            # sampling from composite distribution /  assign next observed state

            for n in range(self.N):
                self.gc[n] = self.mt * self.gc[n] + (1 - self.mt) * cbase[n] * np.mean((-1) ** cls_c[n] * self.V(yc[:, n].T, batch = True))   # momentum for updating c

            self._c = self.project_c(self.c - self.b2 * self.gc, self._S)  # update formula for can-order level

            theta1 = [self.f(x[1], self.s[1]) * self.f(x[2], self.s[2]) * self.f(x[3], self.s[3]),
                      self.f(x[2], self.s[2]) * self.f(x[3], self.s[3]) * (self.f(x[1], self.s[1]) - self.f(x[1], self.c[1])),
                      self.f(x[1], self.s[1]) * self.f(x[3], self.s[3]) * (self.f(x[2], self.s[2]) - self.f(x[2], self.c[2])),
                      self.f(x[1], self.s[1]) * self.f(x[2], self.s[2]) * (self.f(x[3], self.s[3]) - self.f(x[3], self.c[3]))]  # parameter for Multioulli for item1

            e1 = np.random.choice(self.N, p=theta1 / np.sum(theta1), size = self.ms)  # sampling from Multioulli

            ys = np.zeros((self.N, self.N, self.ms))
            for n in range(self.N):
                ys[0, n] = np.where(e1 == n, yc[:, n, n], _x[n] * np.ones(self.ms))

            exp1 = np.where(e1 == 0, cls_c[0], np.where(e1 == 1, cls_c[1], np.where(e1 == 2, cls_c[2], cls_c[3])))

            self.gs[0] = self.mt * self.gs[0] + (1 - self.mt) * sbase[0] * np.mean((-1) ** exp1 * self.V(ys[0], batch=True))

            theta2 = [self.f(x[2], self.s[2]) * self.f(x[3], self.s[3]) * (self.f(x[0], self.s[0]) - self.f(x[0], self.c[0])),
                      self.f(x[0], self.s[0]) * self.f(x[2], self.s[2]) * self.f(x[3], self.s[3]),
                      self.f(x[0], self.s[0]) * self.f(x[3], self.s[3]) * (self.f(x[2], self.s[2]) - self.f(x[2], self.c[2])),
                      self.f(x[0], self.s[0]) * self.f(x[2], self.s[2]) * (self.f(x[3], self.s[3]) - self.f(x[3], self.c[3]))]  # parameter for Multioulli for item2

            e2 = np.random.choice(self.N, p = theta2 / np.sum(theta2), size = self.ms)  # sampling from Multioulli

            for n in range(self.N):
                ys[1, n] = np.where(e2 == n, yc[:, n, n], _x[n] * np.ones(self.ms))
            exp2 = np.where(e2 == 0, cls_c[0], np.where(e2 == 1, cls_c[1], np.where(e2 == 2, cls_c[2], cls_c[3])))

            self.gs[1] = self.mt * self.gs[1] + (1 - self.mt) * sbase[1] * np.mean((-1) ** exp2 * self.V(ys[1], batch = True))

            theta3 = [self.f(x[1], self.s[1]) * self.f(x[3], self.s[3]) * (self.f(x[0], self.s[0]) - self.f(x[0], self.c[0])),
                      self.f(x[0], self.s[0]) * self.f(x[3], self.s[3]) * (self.f(x[1], self.s[1]) - self.f(x[1], self.c[1])),
                      self.f(x[0], self.s[0]) * self.f(x[1], self.s[1]) * self.f(x[3], self.s[3]),
                      self.f(x[0], self.s[0]) * self.f(x[1], self.s[2]) * (self.f(x[3], self.s[3]) - self.f(x[3], self.c[3]))]  # parameter for Multioulli for item3

            e3 = np.random.choice(self.N, p=theta3 / np.sum(theta3), size = self.ms)  # sampling from Multioulli

            for n in range(self.N):
                ys[2, n] = np.where(e3 == n, yc[:, n, n], _x[n] * np.ones(self.ms))
            exp3 = np.where(e3 == 0, cls_c[0], np.where(e3 == 1, cls_c[1], np.where(e3 == 2, cls_c[2], cls_c[3])))

            self.gs[2] = self.mt * self.gs[2] + (1 - self.mt) * sbase[2] * np.mean((-1) ** exp3 * self.V(ys[2], batch=True))

            theta4 = [self.f(x[1], self.s[1]) * self.f(x[2], self.s[2]) * (self.f(x[0], self.s[0]) - self.f(x[0], self.c[0])),
                      self.f(x[0], self.s[0]) * self.f(x[2], self.s[2]) * (self.f(x[1], self.s[1]) - self.f(x[1], self.c[1])),
                      self.f(x[0], self.s[0]) * self.f(x[1], self.s[1]) * (self.f(x[2], self.s[2]) - self.f(x[2], self.c[2])),
                      self.f(x[0], self.s[0]) * self.f(x[1], self.s[1]) * self.f(x[2], self.s[2])]   # parameter for Multioulli for item4

            e4 = np.random.choice(self.N, p=theta4 / np.sum(theta4), size = self.ms)  # sampling from Multioulli

            for n in range(self.N):
                ys[3, n] = np.where(e4 == n, yc[:, n, n], _x[n] * np.ones(self.ms))
            exp4 = np.where(e4 == 0, cls_c[0], np.where(e4 == 1, cls_c[1], np.where(e4 == 2, cls_c[2], cls_c[3])))

            self.gs[3] = self.mt * self.gs[3] + (1 - self.mt) * sbase[3] * np.mean((-1) ** exp4 * self.V(ys[3], batch=True))   # momentum for updating s
            self._s = self.project_s(self.s - self.b3 * self.gs, self._c)   # update formula for reorder level

            self.S = self._S.copy()
            self.c = self._c.copy()
            self.s = self._s.copy()

    def forward(self):
        '''function for conducting forward step for updating agent internal state'''
        self.t += 1  # increment timing

        self.record_s.append(self.s.tolist())
        self.record_c.append(self.c.tolist())
        self.record_S.append(self.S.tolist())
        # store updated policy parameters

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

        self.tau = self.init_tau / (np.int32(self.t / 10) + 1)
        self.sigma = self.init_sigma / (np.int32(self.t / 10) + 1)
        # adaptively decreasing of stochastic factor


    def copy_agent(self, void_agent):
        '''clone validation agent from original agent for validation process (non-stationary)'''
        void_agent.s = self.s.copy()
        void_agent.c = self.c.copy()
        void_agent.S = self.S.copy()
        void_agent.alg_type = self.alg_type
        void_agent.alpha = self.alpha
        void_agent.beta = self.beta
        void_agent.rho = self.rho
        void_agent.cnt = self.cnt.copy()
        void_agent.w = self.w.copy()