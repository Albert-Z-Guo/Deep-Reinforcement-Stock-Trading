import sys

import tensorflow as tf

from utils import *
from agents.DDPG import Agent


if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(
    sys.argv[2]), int(sys.argv[3])
stock_prices = stock_close_prices(stock_name)
l = len(stock_prices) - 1

agent = Agent(state_dim=window_size + 2, initial_funding=10000)

for e in range(1, episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	total_profit = 0
	state = generate_ddpg_state(stock_prices, 0, window_size + 1, agent.balance, len(agent.inventory))

	for t in range(l):
		actions = agent.actor.model.predict(state)
		action = agent.act(state)
		next_state = generate_ddpg_state(stock_prices, t + 1, window_size + 1, agent.balance, len(agent.inventory))
		reward = 0
		loss = 0
		# buy
		if action == 1 and agent.balance > stock_prices[t]:
			agent.balance -= stock_prices[t]
			agent.inventory.append(stock_prices[t])
			print("Buy: " + format_price(stock_prices[t]))
		# sell
		elif action == 2 and len(agent.inventory) > 0:
			agent.balance += stock_prices[t]
			bought_price = agent.inventory.pop(0)
			reward = max(stock_prices[t] - bought_price, 0)
			total_profit += stock_prices[t] - bought_price
			print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(stock_prices[t] - bought_price))
		# hold
		else:
			pass # do nothing

		done = True if t == l - 1 else False
		if done:
			print("--------------------------------")
			print("Total Profit: " + format_price(total_profit))
			print("--------------------------------")

		agent.memory.append((state, actions, reward, next_state, done))
		state = next_state

		if len(agent.memory) > agent.batch_size:
			loss += agent.experience_replay(agent.batch_size, e, t, loss)
			# print("Episode", e, "Step", t, "Action", action, "Reward", reward, "Loss", loss)

	if e % 10 == 0:
		agent.actor.model.save('saved_models/DDPG_ep' + str(e) + '.h5')
		print('model saved')
