import sys

import numpy as np
# np.random.seed(3)  # for reproducible Keras operations
from keras.models import load_model

from utils import *
from agents.DDPG import Agent


if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
initial_funding = 50000
window_size = 10
display = True

agent = Agent(state_dim=13, balance=initial_funding, is_eval=True, model_name=model_name)
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
    actions = agent.act(state, t)
    print('actions:', actions)
    action = np.argmax(actions)
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
