import sys

import numpy as np
# np.random.seed(3) # for reproducible Keras operations

from keras.models import load_model

from utils import *
from agents.DQN import Agent


if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("saved_models/" + model_name)
state_dim = model.layers[0].input.shape.as_list()[1]

agent = Agent(state_dim, True, model_name)
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
batch_size = 32

total_profit = 0
window_size = state_dim
state = generate_state(stock_prices, 0, window_size + 1)

buys = []
sells = []
display = True

for t in range(trading_period):
	action = agent.act(state)
	print(action)
	next_state = generate_state(stock_prices, t + 1, window_size + 1)
	reward = 0

	# buy
	if action == 1:
		agent.inventory.append(stock_prices[t])
		print("Buy: " + format_price(stock_prices[t]))
		buys.append(t)
	# sell
	elif action == 2 and len(agent.inventory) > 0:
		bought_price = agent.inventory.pop(0)
		reward = max(stock_prices[t] - bought_price, 0)
		total_profit += stock_prices[t] - bought_price
		print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(stock_prices[t] - bought_price))
		sells.append(t)
	# hold
	else:
		pass # do nothign

	done = True if t == trading_period - 1 else False
	agent.remember(state, action, reward, next_state, done)
	state = next_state

	if done:
		print("--------------------------------")
		print('{} Total Profit: ${:.2f}'.format(stock_name, total_profit))
		print("--------------------------------")

if display:
	import pandas as pd
	from matplotlib import pyplot as plt

	df = pd.read_csv('./data/{}.csv'.format(stock_name))
	buy_prices = [df.iloc[t, 4] for t in buys]
	sell_prices = [df.iloc[t, 4] for t in sells]

	plt.figure(figsize=(15, 5), dpi=100)
	plt.title('DDPG Total Profit on {}: ${:.2f}'.format(stock_name, total_profit))
	plt.plot(df['Date'], df['Close'], color='black', label=stock_name)
	plt.scatter(buys, buy_prices, c='green', alpha=0.5, label='buy')
	plt.scatter(sells, sell_prices, c='red', alpha=0.5, label='sell')
	plt.xticks(np.linspace(0, len(df), 10))
	plt.ylabel('Price')
	plt.legend()
	plt.grid()
	plt.show()
