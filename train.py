import sys
import argparse

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_funding', action="store", dest="initial_funding", default=50000, type=int, help='episode number')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_funding = inputs.initial_funding

if model_name == 'DQN':
    from agents.DQN import Agent
elif model_name == 'DDPG':
    from agents.DDPG import Agent

agent = Agent(state_dim=window_size + 3, balance=initial_funding)
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0

def buy(t):
    agent.balance -= stock_prices[t]
    agent.inventory.append(stock_prices[t])
    print('Buy: ${:.2f}'.format(stock_prices[t]))


def sell(t):
	agent.balance += stock_prices[t]
	bought_price = agent.inventory.pop(0)
	profit = stock_prices[t] - bought_price
	global reward
	reward = profit
	print('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))

for e in range(1, num_episode + 1):
	print('\nEpisode: {}/{}'.format(e, num_episode))

	agent.reset(initial_funding)
	state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

	for t in range(1, trading_period + 1):
		if t % 100 == 0:
			print('-------------------Period: {}/{}-------------------'.format(t, trading_period))

		reward = 0
		if model_name == 'DQN':
			actions = agent.model.predict(state)[0]
			action = agent.act(state)
		elif model_name == 'DDPG':
			actions = agent.act(state, t)
			action = np.argmax(actions)
		print('actions:', actions, '\n')

		next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
		previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

		# buy
		if action == 1:
			if agent.balance > stock_prices[t]:
				buy(t)
			else:
				reward -= daily_treasury_bond_return_rate() * agent.balance  # missing opportunity
		# sell
		if action == 2:
			if len(agent.inventory) > 0:
				sell(t)
			else:
				reward -= daily_treasury_bond_return_rate() * agent.balance
		# hold
		if action == 0:
			# encourage selling to maximize liquidity
			next_action = np.argsort(actions)[1]
			if next_action == 2 and len(agent.inventory) > 0:
				bought_price = agent.inventory[0]
				profit = stock_prices[t] - bought_price
				if profit > 0:
					sell(t)
				actions[next_action] = 1
			else:
				reward -= daily_treasury_bond_return_rate() * agent.balance

		current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
		unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
		reward += unrealized_profit

		agent.portfolio_values.append(current_portfolio_value)
		agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

		done = True if t == trading_period else False
		if model_name == 'DQN':
			agent.remember(state, action, reward, next_state, done)
		elif model_name == 'DDPG':
			agent.remember(state, actions, reward, next_state, done)

		# update state
		state = next_state

		# experience replay
		if len(agent.memory) > agent.batch_size:
			num_experience_replay += 1
			if model_name == 'DQN':
				loss = agent.experience_replay(agent.batch_size)
			elif model_name == 'DDPG':
				loss = agent.experience_replay(num_experience_replay)
			print('Episode {:.0f} Step {:.0f} Loss {:.2f} Action {:.0f} Reward {:.2f} Balance {:.2f} Number of Stocks {}'.format(e, t, loss, action, reward, agent.balance, len(agent.inventory)))
			agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})

		if done:
			portfolio_return = evaluate_portfolio_performance(agent)
			returns_across_episodes.append(portfolio_return)

    # save models periodically
	if e % 5 == 0:
		if model_name == 'DQN':
			agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
		elif model_name == 'DDPG':
			agent.actor.model.save_weights('saved_models/DDPG_actor_ep' + str(e) + '.h5')
		print('model saved')

plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
