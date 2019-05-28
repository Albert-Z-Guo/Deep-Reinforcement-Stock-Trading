import sys

import numpy as np
np.random.seed(3) # for reproducible Keras operations

from keras.models import load_model

from utils import *
from agents.DDPG import Agent


if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
initial_funding = 50000
batch_size = 32
buys = []
sells = []
return_rates = []
portfolio_values = []
display = True

agent = Agent(state_dim=3, balance=initial_funding, is_eval=True, model_name=model_name)
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1

state = generate_ddpg_state(stock_prices[0], agent.balance, len(agent.inventory))

for t in range(trading_period):
	actions = agent.act(state, t)
	action = np.argmax(actions)
	print(action)
	next_state = generate_ddpg_state(stock_prices[t+1], agent.balance, len(agent.inventory))
	previous_portfolio_value = len(agent.inventory)*stock_prices[t] + agent.balance

	# buy
	if action == 1:
		if agent.balance > stock_prices[t]:
			agent.balance -= stock_prices[t]
			agent.inventory.append(stock_prices[t])
			print("Buy: {}".format(format_price(stock_prices[t])))
			buys.append(t)
	# sell
	elif action == 2:
		if len(agent.inventory) > 0:
			agent.balance += stock_prices[t]
			bought_price = agent.inventory.pop(0)
			profit = stock_prices[t] - bought_price
			print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(profit))
			sells.append(t)
	# hold
	else:
		# print('Hold')
		pass # do nothing

	current_portfolio_value = len(agent.inventory)*stock_prices[t+1] + agent.balance
	return_rates.append((current_portfolio_value-previous_portfolio_value)/previous_portfolio_value)
	portfolio_values.append(current_portfolio_value)
	state = next_state

	done = True if t == trading_period - 1 else False
	if done:
		portfolio_return = current_portfolio_value - initial_funding
		print("--------------------------------")
		print('Portfolio Value: ${:.2f}'.format(current_portfolio_value))
		print('Portfolio Balance: ${:.2f}'.format(agent.balance))
		print('Portfolio Stocks Number: {}'.format(len(agent.inventory)))
		print('{} Return: ${:.2f}'.format(stock_name, portfolio_return))
		print('Mean/Daily Return Rate: {:.3f}%'.format(np.mean(return_rates)*100))
		print('Sharpe Ratio {:.3f}'.format(sharpe_ratio(return_rates)))
		print('Maximum Drawdown {:.3f}%'.format(maximum_drawdown(portfolio_values)*100))
		print("--------------------------------")

if display:
	import pandas as pd
	from matplotlib import pyplot as plt

	df = pd.read_csv('./data/{}.csv'.format(stock_name))
	buy_prices = [df.iloc[t, 4] for t in buys]
	sell_prices = [df.iloc[t, 4] for t in sells]

	plt.figure(figsize=(15, 5), dpi=100)
	plt.title('DDPG Total Return on {}: ${:.2f}'.format(stock_name, portfolio_return))
	plt.plot(df['Date'], df['Close'], color='black', label=stock_name)
	plt.scatter(buys, buy_prices, c='green', alpha=0.5, label='buy')
	plt.scatter(sells, sell_prices, c='red', alpha=0.5, label='sell')
	plt.xticks(np.linspace(0, len(df), 10))
	plt.ylabel('Price')
	plt.legend()
	plt.grid()
	plt.show()
