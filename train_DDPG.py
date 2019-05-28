import sys

import tensorflow as tf

from utils import *
from agents.DDPG import Agent


if len(sys.argv) != 3:
	print("Usage: python train.py [stock] [episodes]")
	exit()


stock_name, episode_count = sys.argv[1], int(sys.argv[2])
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
initial_funding = 50000
agent = Agent(state_dim=3, balance=initial_funding)

for e in range(1, episode_count + 1):
	print('\nEpisode: {}/{}'.format(e, episode_count))

	agent.balance = initial_funding
	agent.inventory = []
	agent.noise.reset()
	total_profit = 0

	state = generate_ddpg_state(stock_prices[0], agent.balance, len(agent.inventory))

	for t in range(trading_period):
		if t % 100 == 0:
			print('-------------------Period: {}/{}-------------------'.format(t+1, trading_period))
		actions = agent.act(state, t)
		action = np.argmax(actions)
		next_state = generate_ddpg_state(stock_prices[t+1], agent.balance, len(agent.inventory))
		reward = 0

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
				reward = stock_prices[t] - bought_price
				total_profit += reward
				print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(stock_prices[t] - bought_price))
		# hold
		else:
			pass # do nothing

		done = True if t == trading_period - 1 else False
		agent.memory.append((state, actions, reward, next_state, done))
		state = next_state

		if len(agent.memory) > agent.batch_size:
			loss = agent.experience_replay(agent.batch_size, e, t)
			print("Episode {:.0f} Step {:.0f} Action {:.0f} Reward {:.2f} Loss {:.2f}".format(e, t, action, reward, loss))

		if done:
			print("------------------------------------------")
			print("Total Profit: " + format_price(total_profit))
			print("------------------------------------------")

	if e % 10 == 0:
		agent.actor.model.save_weights('saved_models/DDPG_actor_ep' + str(e) + '.h5')
		print('model saved')
