import sys

import numpy as np
np.random.seed(1) # for reproducible Keras operations

from keras.models import load_model

from utils import *
from agents.DDPG import Agent


if len(sys.argv) != 3:
	print("Usage: python evaluate.py [stock] [model]")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
# state_dim = load_model("saved_models/" + model_name).layers[0].input.shape.as_list()[1]
state_dim = 3
agent = Agent(state_dim=state_dim, initial_funding=10000, is_eval=True, model_name=model_name)

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
batch_size = 32
total_profit = 0

window_size = state_dim - 2
state = generate_ddpg_state(stock_prices[0], agent.balance, len(agent.inventory))

for t in range(trading_period):
	actions = agent.actor.model.predict(state)
	action = agent.act(state)
	# print(action)
	next_state = generate_ddpg_state(stock_prices[t+1], agent.balance, len(agent.inventory))

	# buy
	if action == 1:
		if agent.balance > stock_prices[t]:
			agent.balance -= stock_prices[t]
			agent.inventory.append(stock_prices[t])
			print("Buy: " + format_price(stock_prices[t]))
	# sell
	elif action == 2:
		if len(agent.inventory) > 0:
			agent.balance += stock_prices[t]
			bought_price = agent.inventory.pop(0)
			total_profit += stock_prices[t] - bought_price
			print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(stock_prices[t] - bought_price))
	# hold
	else:
		# print('Hold')
		pass # do nothing

	state = next_state

	done = True if t == trading_period - 1 else False
	if done:
		print("--------------------------------")
		print(stock_name + " Total Profit: " + format_price(total_profit))
		print("--------------------------------")
