import numpy as np
import matplotlib.pyplot as plt

class Glosten_Milgrom():

    def __init__(self, V_L=100, V_H=200, mu=0.3, sigma=0.5, gamma=0.5, q_0=0.5, t_max=100):
        #
        # parameters
        #
        # model parameters
        self.V_L = V_L
        self.V_H = V_H
        self.mu = mu        # proportion of informed traders
        self.sigma = sigma  # prob. of V_L
        self.gamma = gamma  # prob. of buying for uninformed traders
        self.q_0 = q_0      # market maker's initial belief of V is low (0.5 means no information at all)

        # simulation parameters
        self.t_max = t_max  # maximum number of orders during the simulation
        self.conv = False   # whether the learning process has converged

    def set_env(self):
        #
        # model simulation
        #
        # fundamental value
        self.V = np.random.choice([self.V_L, self.V_H], p=[self.sigma, 1-self.sigma])

        # traders
        self.traders = np.random.choice(["I", "U"], size=self.t_max, p=[self.mu, 1-self.mu])

        # market maker's belief of V is V_L, initialise to q_0
        self.q = [self.q_0, ]

        # some likelihoods designating how informative an observation is
        # note that they depend on the matrix distributions
        self.p_b = np.array([(1-self.mu)*self.gamma, self.mu+(1-self.mu)*self.gamma])  # p(B|V_L), p(B|V_H)
        self.p_s = 1 - self.p_b                                    # p(S|V_L), p(S|V_H)

        # some logging of the process
        self.asks = []       # ask quotes offered by the market maker
        self.bids = []       # bid quotes
        self.prices = []     # transaction prices
        self.profits = []    # profits for the market maker

        # parameters for cauchy convergence test
        self.s = np.maximum(10, self.t_max//10)  # number of terms examined, as s increases, it's more strict
        self.m = 0.003                           # threshold to check against, as m desreases, it's more strict

        # analytics data
        self.analytics = {
            'profit': {
                'I': [0],
                'P': [0],
                'U': [0],
                'M': [0],
            },
            'converged_time': self.t_max,
        }

    def save_analytics(self, **kwargs):
        for key, value in kwargs.items():
            if 'trader_profit' == key:
                trader, profit = value[0], round(value[1])
                for k in self.analytics['profit'].keys():
                    if trader == k:
                        self.analytics['profit'][k].append(profit)
                    else:
                        self.analytics['profit'][k].append(0)
                self.analytics['profit']['M'][-1] = -1 * profit
            elif 'converged_time' == key:
                self.analytics['converged_time'] = int(value)
            else:
                self.analytics[key] = value


    # cauchy convergence test to terminate the simulation if the learning process has converged
    def check_conv(self):
        return np.sum(np.abs(np.diff(self.q[-self.s:]))) < self.m and self.q[-1] != self.q_0

    def gm_model(self):
        # set up the environment for the model
        self.set_env()

        ask, bid = None, None   # starting without any previous quote
        for t in range(self.t_max):
            # market maker's belief of V being V_L or V_H until now
            belief = np.array([self.q[t], 1-self.q[t]])

            #
            # what is the order coming in (buy or sell)
            #
            order = None
            if self.traders[t] == "U":
                order = np.random.choice(["B", "S"], p=[self.gamma, 1-self.gamma])
            elif self.traders[t] in["I", "P"]:
                if self.V == self.V_L:
                    order = "S"
                elif self.V == self.V_H:
                    order = "B"
            #
            # market maker's doing upon receiving an order
            #
            if order == "B":
                # update market maker's belief of V is V_L
                support = self.p_b/np.sum(self.p_b*belief)
                post_belief = support * belief

                # provide the ask (expectation of V according to belief)
                ask = np.sum(np.array([self.V_L, self.V_H]) * post_belief)

                # logging
                self.prices.append(ask)
                self.profits.append(ask - self.V)
                profit = self.V - ask

            elif order == "S":
                # update market maker's belief of V is V_L
                support = self.p_s/np.sum(self.p_s*belief)
                post_belief = support * belief

                # provide the bid (expectation of V according to belief)
                bid = np.sum(np.array([self.V_L, self.V_H]) * post_belief)

                # logging
                self.prices.append(bid)
                self.profits.append(self.V - bid)
                profit = bid - self.V

            # save profit to trader and market maker
            self.save_analytics(trader_profit=(self.traders[t], profit))

            # more logging of the process
            if bid: self.bids.append(bid)
            if ask: self.asks.append(ask)
            self.q.append(post_belief[0])

            # test of convergence, terminate if converges
            if t > self.s and self.check_conv():
                # prompt the convergence status
                self.save_analytics(converged_time=t)
                # print(f'With {t} orders received, the market maker is now {round(self.q[-1]*100, 4)}% confident that V is low.')
                break

    def plot_result(self):
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))
        axs = axs.ravel()
        axs[0].plot(self.q)
        axs[0].set_title('belief on V is low')

        axs[1].plot(self.asks, label = 'ask')
        axs[1].plot(self.bids, label = 'bid')
        axs[1].set_title(f'bid and ask price, true value of V is {self.V}')
        axs[1].legend()

        axs[2].plot(self.prices)
        axs[2].set_title('transaction price')

        axs[3].plot([a-b for a, b in zip(self.asks, self.bids)])
        axs[3].set_title('bid-ask spread')

        # expected profit over time
        exp_pnl = [x/(i+1) for i, x in enumerate(np.cumsum(self.profits))]
        axs[4].plot(exp_pnl)
        axs[4].set_title('moving average profit over orders received')

        axs[5].set_visible(False)

        plt.tight_layout()

    def plot_profit(self, star=False):
        full_trader = {
            'I': 'Informed',
            'P': 'Partially informed',
            'U': 'Uninformed',
            'M': 'Market maker',
        }
        for trader in self.analytics['profit']:
            profits = self.analytics['profit'][trader]
            if len(set(profits)) == 1 and profits[0] == 0:
                continue
            elif star:
                plt.plot(np.cumsum(profits), '-*', label=full_trader[trader])
            else:
                plt.plot(np.cumsum(profits), label=full_trader[trader])
        plt.xlabel(f"orders ({self.analytics['converged_time']})")
        plt.ylabel('profit')
        plt.title(f"Welfare($\sigma={self.sigma}, \mu={self.mu}, \gamma={self.gamma}$)")
        plt.legend()