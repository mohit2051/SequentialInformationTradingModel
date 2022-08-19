import numpy as np
import matplotlib.pyplot as plt
from glosten_milgrom_base import Glosten_Milgrom
import argparse


class GMchild(Glosten_Milgrom):

    def __init__(self, V_L=100, V_H=200, mu=0.3, sigma=0.5, gamma=0.5, q_0=0.5, t_max=100, theta=0):
        super().__init__()
        self.V_L = V_L
        self.V_H = V_H
        self.mu = mu          # proportion of informed traders
        self.sigma = sigma    # prob. of V_L
        self.gamma = gamma    # prob. of buying for uninformed traders
        self.q_0 = q_0
        self.t_max = t_max
        self.theta = theta    # probability if the uninformed trader will remain active
        self.UninformedMakeTrade = None
    
    def set_env(self):
        super().set_env()
        """
        Put here whatever you wish to overwrite in the set_env() function.
        """
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
        self.p_b = np.array([(1-self.mu)*self.gamma*self.theta, self.mu+(1-self.mu)*self.gamma*self.theta]) # p(B|V_L), p(B|V_H)
        self.p_s = np.array([self.mu +(1-self.mu)*(1-self.gamma)*self.theta, (1-self.mu)*(1-self.gamma)*self.theta])              # p(S|V_L), p(S|V_H)

        # some logging of the process
        self.asks = []       # ask quotes offered by the market maker
        self.bids = []       # bid quotes
        self.prices = []     # transaction prices
        self.profits = []    # profits for the market maker

        # parameters for cauchy convergence test
        self.s = np.maximum(10, self.t_max//10)  # as s increases, it's more strict
        self.m = 0.003                           # as m desreases, it's more strict

        del self.analytics['profit']['P']

    def gm_model(self):
        # set up the environment for the model
        self.set_env()

        ask, bid = None, None   # starting without any previous quote
        count = 0
        for t in range(self.t_max):
            # market maker's belief of V being V_L or V_H until now
            belief = np.array([self.q[t], 1-self.q[t]])

            
            
            #
            # what is the order coming in (buy or sell)
            #
            order = None
            if self.traders[t] == "U":
                #check if uninformed trader will make trade
                self.UninformedMakeTrade = np.random.choice([1,0], p = [self.theta, (1-self.theta)])
                if(self.UninformedMakeTrade == 1):
                    order = np.random.choice(["B", "S"], p=[self.gamma, (1-self.gamma)])
                else:
                    order = "No Order"

            elif self.traders[t] == "I":
                if self.V == self.V_L:
                    order = "S"
                elif self.V == self.V_H:
                    order = "B"

            #
            # market maker's doing upon receiving an order
            #
            count+=1
            #print(f"order = {order} and count = {count}", end = "\n")
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
            
            elif order == "No Order":
                #this basically implies an inactive trader
                #now p_b and p_s will be updated only with Informed traders buy or sell respectively
                #self.p_b = np.array([0, self.mu])  # p(B|V_L), p(B|V_H)
                #self.p_s = 1 - self.p_b            # p(S|V_L), p(S|V_H) 
                post_belief = belief
                profit = 0

                #market maker has learning from the fact that he knows that this is an uninformed trader who is inactive

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
                #print(f'With {t} orders received, the market maker is now {round(self.q[-1]*100, 4)}% confident that V is low.')
                break
    def plot_profit(self, star=False):
        super().plot_profit(star=star)
        plt.title(f"Welfare($\sigma={self.sigma}, \mu={self.mu}, \gamma={self.gamma}, \\theta={self.theta}$)")

def saveimg(path='.', filename=None, plt=None):
    filepath = f'{path}/v2_{filename}'
    plt.savefig(filepath)
    plt.clf()
    print(f'- {filename} is saved!')

def plot_efficiencyvstheta(theta_list):

    """
    This method will plot graph of Number of Orders Market Maker recieved to identify the true value of asset against theta
    It will be shown using a series of boxplot
    """   
    final_prices = list()
    # Iterate through each theta in list
    
    for i in range(len(theta_list)):
        prices_t = list()
        #perfoming 50 iterations against each theta to compute the numbers of order required by Market Maker to indetify the true value of the asset
        for _ in range(50):
            model = GMchild(theta=theta_list[i])
            model.gm_model()
            price = len(model.prices)
            prices_t.append(price) 
        final_prices.append(prices_t)

    plt.boxplot(final_prices)
    plt.xticks(list(range(1,len(theta_list)+1)), np.round(theta_list,1))
    plt.xlabel("theta")
    plt.ylabel("Orders Recieved by Market Maker")
    plt.title(f'Learning Efficiency with varying theta')
    saveimg(path=args.path, filename='efficiency_theta', plt=plt)

def plot_profit_distribution():
    """
    This method will plot graph of distribution of welfare between the three agents i.e., the Market Maker, Informed Trader, and the Uninformed Trader
    """ 
    std_threshold = 2
    thetalst = [0.2, 0.6, 0.9]
    final_plot_list = list()
    for i in range(len(thetalst)):
        #print(f"theta = {theta_list[i]}")
        new_dict = dict()
        new_dict["I"] = list()
        new_dict["U"] = list()
        new_dict["M"] = list()

        #Performing 1000 simulations for computing profits of each agent
        for _ in range(1000):
            model = GMchild(theta=thetalst[i])
            model.gm_model()

            x = model.analytics["profit"]
            for k,v in x.items():
                final_profit = np.cumsum(v)[-1]
                new_dict[k].append(final_profit)
        
        #remove outliers
        dis = np.array(list(new_dict.values())).ravel()
        std = [np.mean(dis) - std_threshold * np.std(dis), np.mean(dis) + std_threshold * np.std(dis)]
        for key, value in new_dict.items():
                new_dict[key] = [v for v in value if v >= std[0] and v <= std[1]]

        final_plot_list.append(new_dict)

    fig, axs = plt.subplots(ncols=1, nrows=3, figsize = (15,12), sharex = True, sharey=True)

    #Setting up axes for 3 plots with respect to theta and for each theta value distribution of profit between Market Maker, Informed Trader and Uninformed Trader
    for i in range(3):
        axs[i].hist(final_plot_list[i]["M"], histtype="step", bins=70, label =f"Market Maker, mean = {np.mean(final_plot_list[i]['M'])}")
        axs[i].hist(final_plot_list[i]["I"], histtype="step", bins=70, label=f"Informed Trader,  mean = {np.mean(final_plot_list[i]['I'])}")
        axs[i].hist(final_plot_list[i]["U"], histtype="step", bins=70, label=f"Uninformed Trader,  mean = {np.mean(final_plot_list[i]['U'])}")
        axs[i].set_title(f"$\\theta = {thetalst[i]}$")
        axs[i].axvline(x=0, color='red', linestyle='dashed', linewidth=1.5, alpha=0.4, label='zero profit')
        axs[i].legend(loc='upper right')
    
    plt.tight_layout()
    fig.text(0.5, 0.001, "Profits", ha="center", va="bottom", fontsize=15)
    saveimg(path=args.path, filename='welfare_theta', plt=plt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to save images", type=str, default=".")
    parser.add_argument("-a", "--all", help="path to save images", action="store_true")
    parser.add_argument("--result", help="plot result", action="store_true")
    parser.add_argument("--efficiency", help="plot efficiency vs theta", action="store_true")
    parser.add_argument("--distribution", help="plot distribution", action="store_true")
    args = parser.parse_args()

    #
    # The use of the Glosten_Milgrom class itself.
    #
    # Note that the there are default values for the parameters, but you can pass your own value as well.

    theta_list = np.arange(0.1,1.1,0.1)
    t_max = 100

    if args.all or args.result:
        your_model = GMchild(V_L=100, V_H=200, mu=0.3, sigma=0.5, gamma=0.5, q_0=0.5, t_max=200, theta=1)
        your_model.gm_model()
        your_model.plot_result()
        saveimg(path=args.path, filename='results', plt=plt)

    
    if args.all or args.efficiency:
        plot_efficiencyvstheta(theta_list)

    if args.all or args.distribution:
        plot_profit_distribution()

    else:
        raise ValueError("Valid Plot option not selected")

   
    

    