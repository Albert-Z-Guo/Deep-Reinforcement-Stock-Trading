import sys

import numpy as np
np.random.seed(3)  # for reproducible Keras operations
from keras.models import load_model

from utils import *
from agents.DQN import Agent


if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
initial_funding = 50000
model = load_model("saved_models/" + model_name)
state_dim = model.layers[0].input.shape.as_list()[1]

agent = Agent(state_size=state_dim, balance=initial_funding, is_eval=True, model_name=model_name)
stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
display = True

window_size = state_dim
state = generate_state(stock_prices, 0, window_size + 1)


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
    actions = agent.model.predict(state)[0]
    print('actions:' actions)
    action = agent.act(state)
    print('chosen action:' action)
    next_state = generate_state(stock_prices, t, window_size + 1)
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
