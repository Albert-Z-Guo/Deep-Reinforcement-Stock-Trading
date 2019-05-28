import numpy as np


def format_price(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def stock_close_prices(key):
	'''return a list containing stock close prices from a .csv file'''
	prices = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()
	for line in lines[1:]:
		prices.append(float(line.split(",")[4]))
	return prices


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def generate_state(stock_prices, t, n):
	'''
	return an n-day state representation ending at time t
	the state is defined as the adjacent daily stock price differences (sigmoid)
	for a n-day period
	'''
	d = t - n + 1
	block = stock_prices[d:t + 1] if d >= 0 else -d * [stock_prices[0]] + stock_prices[0:t + 1] # pad with t_0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))
	return np.array([res])


def generate_ddpg_state(stock_price, balance, num_holding):
	return np.array([[stock_price, balance, num_holding]])


def daily_risk_free_interest_rate():
	r_year = 2.75 / 100 # approximate annual U.S. Treasury bond return rate
	return (1 + r_year)**(1/365) - 1


def sharpe_ratio(return_rates):
	risk_free_rate = daily_risk_free_interest_rate()
	numerator = np.mean(np.array(return_rates) - risk_free_rate)
	denominator = np.std(np.array(return_rates) - risk_free_rate)
	return numerator / denominator


# reference: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_actions(self, actions, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(actions + ou_state, 0, 1)
