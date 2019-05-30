from agents.DDPG import Agent
from utils import *
from keras.models import load_model
import sys

import numpy as np
np.random.seed(3)  # for reproducible Keras operations


if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
initial_funding = 50000
batch_size = 32
display = True

agent = Agent(state_dim=3, balance=initial_funding, is_eval=True, model_name=model_name)
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1

state = generate_ddpg_state(stock_prices[0], agent.balance, len(agent.inventory))

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
    reward = max(profit, 0)
    agent.sell_dates.append(t)
    print('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))

for t in range(trading_period):
    actions = agent.act(state, t)
    action = np.argmax(actions)
    print(action)
    next_state = generate_ddpg_state(stock_prices[t + 1], agent.balance, len(agent.inventory))
    previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

	# buy
    if action == 1:
        if agent.balance > stock_prices[t]: buy(t)
    else:
        next_action = np.argsort(actions)[1]  # second predicted action
        if next_action == 2: sell(t)
	# sell
    if action == 2:
        if len(agent.inventory) > 0:
            sell(t)
        else:
            next_action = np.argsort(actions)[1]
            if next_action == 1: buy(t)
    # hold

    current_portfolio_value = len(agent.inventory) * stock_prices[t + 1] + agent.balance
    agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
    agent.portfolio_values.append(current_portfolio_value)
    state = next_state

    done = True if t == trading_period - 1 else False
    if done:
        portfolio_return = evaluate_portfolio_performance(agent)

if display:
    plot_portfolio_transaction_history(stock_name, agent)
    plot_portfolio_value_comparison(stock_name, agent)
