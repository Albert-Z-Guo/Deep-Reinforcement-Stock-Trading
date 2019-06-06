import sys
import argparse

import numpy as np
np.random.seed(3)  # for reproducible Keras operations

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DQN_ep1.h5', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2018', help="stock name")
parser.add_argument('--initial_funding', action="store", dest="initial_funding", default=50000, help='episode number')
inputs = parser.parse_args()

model_name = inputs.model_name
model_to_load = inputs.model_to_load
stock_name = inputs.stock_name
initial_funding = inputs.initial_funding
display = True
window_size = 10

if model_name == 'DQN':
	from agents.DQN import Agent
elif model_name == 'DDPG':
	from agents.DDPG import Agent

agent = Agent(state_dim=13, balance=initial_funding, is_eval=True, model_name=model_to_load)
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1

state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

def buy(t):
    agent.balance -= stock_prices[t]
    agent.inventory.append(stock_prices[t])
    agent.buy_dates.append(t)
    print('Buy: ${:.2f}'.format(stock_prices[t]))

def sell(t):
    agent.balance += stock_prices[t]
    bought_price = agent.inventory.pop(0)
    profit = stock_prices[t] - bought_price
    global reward
    reward = profit
    agent.sell_dates.append(t)
    print('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))

for t in range(1, trading_period + 1):
    if model_name == 'DQN':
        actions = agent.model.predict(state)[0]
        action = agent.act(state)
    elif model_name == 'DDPG':
        actions = agent.act(state, t)
        action = np.argmax(actions)

    print('actions:', actions)
    print('chosen action:', action)

    next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
    previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

    # buy
    if action == 1 and agent.balance > stock_prices[t]: buy(t)
    # sell
    if action == 2 and len(agent.inventory) > 0: sell(t)
    # hold
    if action == 0: pass

    current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
    agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
    agent.portfolio_values.append(current_portfolio_value)
    state = next_state

    done = True if t == trading_period else False
    if done:
        portfolio_return = evaluate_portfolio_performance(agent)

if display:
    # plot_portfolio_transaction_history(stock_name, agent)
    # plot_portfolio_performance_comparison(stock_name, agent)
    plot_all(stock_name, agent)
