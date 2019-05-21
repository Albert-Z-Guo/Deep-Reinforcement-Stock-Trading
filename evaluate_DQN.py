import sys

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
l = len(stock_prices) - 1
batch_size = 32

total_profit = 0
agent.inventory = []
window_size = state_dim
state = generate_state(stock_prices, 0, window_size + 1)

for t in range(l):
	action = agent.act(state)

	next_state = generate_state(stock_prices, t + 1, window_size + 1)
	reward = 0

	# buy
	if action == 1:
		agent.inventory.append(stock_prices[t])
		print("Buy: " + format_price(stock_prices[t]))
	# sell
	elif action == 2 and len(agent.inventory) > 0:
		bought_price = agent.inventory.pop(0)
		reward = max(stock_prices[t] - bought_price, 0)
		total_profit += stock_prices[t] - bought_price
		print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(stock_prices[t] - bought_price))
	# hold
	else:
		pass # do nothign

	done = True if t == l - 1 else False
	agent.remember(state, action, reward, next_state, done)
	state = next_state

	if done:
		print("--------------------------------")
		print(stock_name + " Total Profit: " + format_price(total_profit))
		print("--------------------------------")
