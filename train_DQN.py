import sys

from utils import *
from agents.DQN import Agent


if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
batch_size = 32
initial_funding = 50000

agent = Agent(window_size, balance=initial_funding)

for e in range(1, episode_count + 1):
	print('\nEpisode: {}/{}'.format(e, episode_count))
	state = generate_state(stock_prices, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(trading_period):
		if t % 100 == 0:
			print('-------------------Period: {}/{}-------------------'.format(t+1, trading_period))
		action = agent.act(state)
		next_state = generate_state(stock_prices, t + 1, window_size + 1)
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
				reward = max(stock_prices[t] - bought_price, 0)
				total_profit += stock_prices[t] - bought_price
				print("Sell: " + format_price(stock_prices[t]) + " | Profit: " + format_price(stock_prices[t] - bought_price))
		# hold
		else:
			pass # do nothing

		done = True if t == trading_period - 1 else False
		agent.remember(state, action, reward, next_state, done)
		state = next_state

		if len(agent.memory) > batch_size:
			agent.experience_replay(batch_size)

		if done:
			print("------------------------------------------")
			print("Total Profit: " + format_price(total_profit))
			print("------------------------------------------")

	if e % 1 == 0:
		agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
		print('model saved')
