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
returns_across_episodes  = []

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

for e in range(1, episode_count + 1):
    print('\nEpisode: {}/{}'.format(e, episode_count))

    agent.reset(initial_funding)
    state = generate_ddpg_state(stock_prices[0], agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            print('-------------------Period: {}/{}-------------------'.format(t, trading_period))

        reward = 0
        actions = agent.act(state, t)
        print(actions)
        action = np.argmax(actions)

        next_state = generate_ddpg_state(stock_prices[t], agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

		# buy
        if action == 1:
            if agent.balance > stock_prices[t]: buy(t)
            else: reward -= daily_treasury_bond_return_rate() * agent.balance # missing opportunity
        # sell
        if action == 2:
            if len(agent.inventory) > 0: sell(t)
            else: reward -= daily_treasury_bond_return_rate() * agent.balance
	    # hold
        if action == 0:
            # encourage selling to maximize liquidity
            next_action = np.argsort(actions)[1]
            if next_action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory[0]
                profit = stock_prices[t] - bought_price
                if profit > 0: sell(t)
                actions[next_action] = 1
            else:
                reward -= daily_treasury_bond_return_rate() * agent.balance

        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        agent.portfolio_values.append(current_portfolio_value)

        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)
        state = next_state

        if len(agent.memory) > agent.batch_size:
            loss = agent.experience_replay(e, t)
            print("Episode {:.0f} Step {:.0f} Action {:.0f} Reward {:.2f} Loss {:.2f}".format(e, t, action, reward, loss))

        if done:
            portfolio_return = evaluate_portfolio_performance(agent)
            returns_across_episodes.append(portfolio_return)

    if e % 1 == 0:
        agent.actor.model.save_weights('saved_models/DDPG_actor_ep' + str(e) + '.h5')
        print('model saved')

plot_portfolio_returns_across_episodes(returns_across_episodes)
