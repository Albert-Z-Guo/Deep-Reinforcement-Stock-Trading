import numpy as np
import pandas as pd
from empyrical import sharpe_ratio
from matplotlib import pyplot as plt


class Portfolio:
    def __init__(self, balance=50000):
        self.initial_portfolio_value = balance
        self.balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [balance]
        self.buy_dates = []
        self.sell_dates = []

    def reset_portfolio(self):
        self.balance = self.initial_portfolio_value
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [self.initial_portfolio_value]

        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def stock_close_prices(key):
    '''return a list containing stock close prices from a .csv file'''
    prices = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        prices.append(float(line.split(",")[4]))
    return prices


def generate_price_state(stock_prices, end_index, window_size):
    '''
    return a state representation, defined as
    the adjacent stock price differences after sigmoid function (for the past window_size days up to end_date)
    note that a state has length window_size, a period has length window_size+1
    '''
    start_index = end_index - window_size
    if start_index >= 0:
        period = stock_prices[start_index:end_index+1]
    else: # if end_index cannot suffice window_size, pad with prices on start_index
        period = -start_index * [stock_prices[0]] + stock_prices[0:end_index+1]
    return sigmoid(np.diff(period))


def generate_portfolio_state(stock_price, balance, num_holding):
    '''logarithmic values of stock price, portfolio balance, and number of holding stocks'''
    return [np.log(stock_price), np.log(balance), np.log(num_holding + 1e-6)]


def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
    '''
    return a state representation, defined as
    adjacent stock prices differences after sigmoid function (for the past window_size days up to end_date) plus
    logarithmic values of stock price at end_date, portfolio balance, and number of holding stocks
    '''
    prince_state = generate_price_state(stock_prices, end_index, window_size)
    portfolio_state = generate_portfolio_state(stock_prices[end_index], balance, num_holding)
    return np.array([np.concatenate((prince_state, portfolio_state), axis=None)])


def treasury_bond_daily_return_rate():
    r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
    return (1 + r_year)**(1 / 365) - 1


def maximum_drawdown(portfolio_values):
    end_index = np.argmax(np.maximum.accumulate(portfolio_values) - portfolio_values)
    if end_index == 0:
        return 0
    beginning_iudex = np.argmax(portfolio_values[:end_index])
    return (portfolio_values[end_index] - portfolio_values[beginning_iudex]) / portfolio_values[beginning_iudex]


def evaluate_portfolio_performance(agent, logger):
    portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    logger.info("--------------------------------")
    logger.info('Portfolio Value:        ${:.2f}'.format(agent.portfolio_values[-1]))
    logger.info('Portfolio Balance:      ${:.2f}'.format(agent.balance))
    logger.info('Portfolio Stocks Number: {}'.format(len(agent.inventory)))
    logger.info('Total Return:           ${:.2f}'.format(portfolio_return))
    logger.info('Mean/Daily Return Rate:  {:.3f}%'.format(np.mean(agent.return_rates) * 100))
    logger.info('Sharpe Ratio adjusted with Treasury bond daily return: {:.3f}'.format(sharpe_ratio(np.array(agent.return_rates)), risk_free=treasury_bond_daily_return_rate()))
    logger.info('Maximum Drawdown:        {:.3f}%'.format(maximum_drawdown(agent.portfolio_values) * 100))
    logger.info("--------------------------------")
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
	# compare with S&P 500 performance in 2018
	if '^GSPC' not in stock_name:
		dates, GSPC_buy_and_hold_portfolio_values, GSPC_buy_and_hold_return = buy_and_hold_benchmark('^GSPC_2018', agent)
		plt.plot(dates, GSPC_buy_and_hold_portfolio_values, color='red', label='S&P 500 2018 Buy and Hold Total Return: ${:.2f}'.format(GSPC_buy_and_hold_return))
	plt.xticks(np.linspace(0, len(dates), 10))
	plt.ylabel('Portfolio Value ($)')
	plt.legend()
	plt.grid()
	plt.show()


def plot_all(stock_name, agent):
    '''combined plots of plot_portfolio_transaction_history and plot_portfolio_performance_comparison'''
    fig, ax = plt.subplots(2, 1, figsize=(16,8), dpi=100)

    portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    df = pd.read_csv('./data/{}.csv'.format(stock_name))
    buy_prices = [df.iloc[t, 4] for t in agent.buy_dates]
    sell_prices = [df.iloc[t, 4] for t in agent.sell_dates]
    ax[0].set_title('{} Total Return on {}: ${:.2f}'.format(agent.model_type, stock_name, portfolio_return))
    ax[0].plot(df['Date'], df['Close'], color='black', label=stock_name)
    ax[0].scatter(agent.buy_dates, buy_prices, c='green', alpha=0.5, label='buy')
    ax[0].scatter(agent.sell_dates, sell_prices,c='red', alpha=0.5, label='sell')
    ax[0].set_ylabel('Price')
    ax[0].set_xticks(np.linspace(0, len(df), 10))
    ax[0].legend()
    ax[0].grid()

    dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(stock_name, agent)
    agent_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    ax[1].set_title('{} vs. Buy and Hold'.format(agent.model_type))
    ax[1].plot(dates, agent.portfolio_values, color='green', label='{} Total Return: ${:.2f}'.format(agent.model_type, agent_return))
    ax[1].plot(dates, buy_and_hold_portfolio_values, color='blue', label='{} Buy and Hold Total Return: ${:.2f}'.format(stock_name, buy_and_hold_return))
    # compare with S&P 500 performance in 2018 if stock is not S&P 500
    if '^GSPC' not in stock_name:
    	dates, GSPC_buy_and_hold_portfolio_values, GSPC_buy_and_hold_return = buy_and_hold_benchmark('^GSPC_2018', agent)
    	ax[1].plot(dates, GSPC_buy_and_hold_portfolio_values, color='red', label='S&P 500 2018 Buy and Hold Total Return: ${:.2f}'.format(GSPC_buy_and_hold_return))
    ax[1].set_ylabel('Portfolio Value ($)')
    ax[1].set_xticks(np.linspace(0, len(df), 10))
    ax[1].legend()
    ax[1].grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_portfolio_returns_across_episodes(model_name, returns_across_episodes):
    len_episodes = len(returns_across_episodes)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.title('Portfolio Returns')
    plt.plot(returns_across_episodes, color='black')
    plt.xlabel('Episode')
    plt.ylabel('Return Value')
    plt.grid()
    plt.savefig('visualizations/{}_returns_ep{}.png'.format(model_name, len_episodes))
    plt.show()
