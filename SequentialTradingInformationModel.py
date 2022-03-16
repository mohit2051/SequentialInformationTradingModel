
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class GlostenMilgrom:

    def __init__(self):
        pass
    def get_V(self, V_h : int, V_l : int) -> int:
        """
        This method returns the value of V with equal probabilitiies of choice
        """
        return( np.random.choice(np.array([V_h,V_l])))

    def get_I_U(self, mu):
        """
        This method returns the value of V with equal probabilitiies of choice
        """
        return(random.choices(["I","U"],weights = [mu, 1-mu], k=t_max))

    def get_B_S_list(self, gamma, V, V_h, traderType_lst):
        BS_lst = list()
        for traderType in traderType_lst: 
            if(V == V_h):
                if(traderType == "I"):
                    BS_lst.append("B")
                    
                else:
                    BS_lst.extend(random.choices(["B","S"],weights = [gamma, 1-gamma], k=1))
            else:
                if(traderType == "I"):
                    BS_lst.append("S")
                    
                else:
                    BS_lst.extend(random.choices(["B","S"],weights = [gamma, 1-gamma], k=1))
        return BS_lst

if __name__ =="__main__":
    V_l, V_h = 100, 200
    mu, delta = 0.3, 0.5
    t_max = 100
    
    gm  = GlostenMilgrom()

    V = gm.get_V(V_l, V_h)
    print(f"Value of V is {V} and type of V is {type(V)}")
    
    traderType_lst=gm.get_I_U(mu)
    print(f"Type of trader list = {traderType_lst}")

    gamma = 0.5
    BS_lst = gm.get_B_S_list(gamma,V, V_h, traderType_lst)
    
    print("\n")
    print(f"Buy and Sell list = ", BS_lst)
    print("\n")

    ask_prices = list()
    bid_prices = list()

    prices_lst = list()
    profits_lst = list()

    p_vl = delta
    for value in BS_lst:
        if(value == "B"):
            p_b_vh = mu + (1-mu)*gamma
            p_b_vl = (1-mu)*gamma
            p_b = (1-p_vl)*p_b_vh + p_vl*p_b_vl
            p_vl_b = (p_b_vl * p_vl)/p_b 

            a = V_h*(1-p_vl_b) + V_l*p_vl_b
            ask_prices.append(a)

            p_s_vh = (1-mu) * (1-gamma)
            p_s_vl = mu + (1-mu)*(1-gamma)
            p_s = (1-p_vl)*p_s_vh + p_vl*p_s_vl
            p_vl_s = (p_s_vl * p_vl)/p_s

            b = V_h * (1-p_vl_s) + V_l * p_vl_s
            bid_prices.append(b)

            prices_lst.append(a)
            profit = a - V
            profits_lst.append(profit)

            p_vl = p_vl_b

        elif(value == "S"):
            p_s_vh = (1-mu) * (1-gamma)
            p_s_vl = mu + (1-mu)*(1-gamma)
            p_s = (1-p_vl)*p_s_vh + p_vl*p_s_vl
            p_vl_s = (p_s_vl * p_vl)/p_s

            b = V_h * (1-p_vl_s) + V_l * p_vl_s
            bid_prices.append(b)

            p_b_vh = mu + (1-mu)*gamma
            p_b_vl = (1-mu)*gamma
            p_b = (1-p_vl)*p_b_vh + p_vl*p_b_vl
            p_vl_b = (p_b_vl * p_vl)/p_b 

            a = V_h*(1-p_vl_b) + V_l*p_vl_b
            ask_prices.append(a)

            prices_lst.append(b)
            profit = V - b
            profits_lst.append(profit)

            p_vl = p_vl_s

    spread_lst = [a_price - b_price for a_price, b_price in zip(ask_prices,bid_prices) ]
    print(f"spread of list = {spread_lst}")
    time_lst = np.arange(0,t_max,1)
    final_DF = pd.DataFrame(np.column_stack([bid_prices,ask_prices,prices_lst,spread_lst, profits_lst,time_lst ]), columns = ["Bids", "Asks", "Prices", "Spread", "Profit", "time"])
    print(final_DF)
    final_DF.plot(x = "time", y = ["Bids","Asks","Prices"])
    plt.show()

    
