import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# reference: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_actions(self, actions, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(actions + ou_state, 0, 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def stock_close_prices(key):
    '''return a list containing stock close prices from a .csv file'''
    prices = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        prices.append(float(line.split(",")[4]))
    return prices


def generate_state(stock_prices, t, n):
    '''
    return an n-day state representation ending at time t
    the state is defined as the adjacent daily stock price differences (sigmoid)
    for a n-day period
    '''
    d = t - n + 1
    block = stock_prices[d:t + 1] if d >= 0 else -d * [stock_prices[0]] + stock_prices[0:t + 1]  # pad with t_0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


def generate_ddpg_state(stock_price, balance, num_holding):
    return np.array([[stock_price, balance, num_holding]])


def daily_treasury_bond_return_rate():
    r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
    return (1 + r_year)**(1 / 365) - 1


# reference: https://en.wikipedia.org/wiki/Sharpe_ratio
def sharpe_ratio(return_rates):
	'''ex-ante Sharpe ratio'''
	risk_free_rate = daily_treasury_bond_return_rate()
	numerator = np.mean(np.array(return_rates) - risk_free_rate)
	denominator = np.std(np.array(return_rates) - risk_free_rate)
	if denominator == 0: # invalid case
		return 0
	return numerator / denominator


def maximum_drawdown(portfolio_values):
    end_index = np.argmax(np.maximum.accumulate(portfolio_values) - portfolio_values)
    if end_index == 0:
        return 0
    beginning_iudex = np.argmax(portfolio_values[:end_index])
    return (portfolio_values[end_index] - portfolio_values[beginning_iudex]) / portfolio_values[beginning_iudex]


def evaluate_portfolio_performance(agent):
    current_portfolio_value = agent.portfolio_values[-1]
    portfolio_return = current_portfolio_value - agent.initial_portfolio_value
    print("--------------------------------")
    print('Portfolio Value: ${:.2f}'.format(current_portfolio_value))
    print('Portfolio Balance: ${:.2f}'.format(agent.balance))
    print('Portfolio Stocks Number: {}'.format(len(agent.inventory)))
    print('Total Return: ${:.2f}'.format(portfolio_return))
    print('Mean/Daily Return Rate: {:.3f}%'.format(np.mean(agent.return_rates) * 100))
    print('Sharpe Ratio: {:.3f}'.format(sharpe_ratio(agent.return_rates)))
    print('Maximum Drawdown: {:.3f}%'.format(maximum_drawdown(agent.portfolio_values) * 100))
    print("--------------------------------")
    return portfolio_return


def plot_portfolio_transaction_history(stock_name, agent):
	portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
	df = pd.read_csv('./data/{}.csv'.format(stock_name))
	buy_prices = [df.iloc[t, 4] for t in agent.buy_dates]
	sell_prices = [df.iloc[t, 4] for t in agent.sell_dates]
	plt.figure(figsize=(15, 5), dpi=100)
	plt.title('{} Total Return on {}: ${:.2f}'.format(agent.model_type, stock_name, portfolio_return))
	plt.plot(df['Date'], df['Close'], color='black', label=stock_name)
	plt.scatter(agent.buy_dates, buy_prices, c='green', alpha=0.5, label='buy')
	plt.scatter(agent.sell_dates, sell_prices,c='red', alpha=0.5, label='sell')
	plt.xticks(np.linspace(0, len(df), 10))
	plt.ylabel('Price')
	plt.legend()
	plt.grid()
	plt.show()


def buy_and_hold_benchmark(stock_name, agent):
	df = pd.read_csv('./data/{}.csv'.format(stock_name))
	dates = df['Date']
	num_holding = agent.initial_portfolio_value // df.iloc[0, 4]
	balance_left = agent.initial_portfolio_value % df.iloc[0, 4]
	buy_and_hold_portfolio_values = df['Close']*num_holding + balance_left
	buy_and_hold_return = buy_and_hold_portfolio_values.iloc[-1] - agent.initial_portfolio_value
	return dates, buy_and_hold_portfolio_values, buy_and_hold_return


def plot_portfolio_performance_comparison(stock_name, agent):
	dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(stock_name, agent)
	agent_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
	plt.figure(figsize=(15, 5), dpi=100)
	plt.title('{} vs. Buy and Hold'.format(agent.model_type))
	plt.plot(dates, agent.portfolio_values, color='green', label='{} Total Return: ${:.2f}'.format(agent.model_type, agent_return))
	plt.plot(dates, buy_and_hold_portfolio_values, color='blue', label='{} Buy and Hold Total Return: ${:.2f}'.format(stock_name, buy_and_hold_return))
	# compare with S&P 500 2018
	if '^GSPC' not in stock_name:
		dates, GSPC_buy_and_hold_portfolio_values, GSPC_buy_and_hold_return = buy_and_hold_benchmark('^GSPC_2018', agent)
		plt.plot(dates, GSPC_buy_and_hold_portfolio_values, color='red', label='S&P 500 2018 Buy and Hold Total Return: ${:.2f}'.format(GSPC_buy_and_hold_return))
	plt.xticks(np.linspace(0, len(dates), 10))
	plt.ylabel('Portfolio Value ($)')
	plt.legend()
	plt.grid()
	plt.show()


def plot_portfolio_returns_across_episodes(returns_across_episodes):
    plt.figure(figsize=(15, 5), dpi=100)
    plt.title('Portfolio Returns')
    plt.plot(returns_across_episodes, color='black')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.grid()
    plt.show()
