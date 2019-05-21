import sys

import tensorflow as tf

from utils import *
from agents.DDPG import Agent


if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stock_prices = stock_close_prices(stock_name)
l = len(stock_prices) - 1

# Tensorflow GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

agent = Agent(window_size + 2, sess)


for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    total_profit = 0
	# skip random process for action exploration
    state = generate_ddpg_state(stock_prices, 0, window_size + 1, total_profit, len(agent.inventory))

    for t in range(l):
        actions = agent.actor.model.predict(state)
        action = agent.act(state)
        next_state = generate_ddpg_state(stock_prices, t + 1, window_size + 1, total_profit, len(agent.inventory))
        reward = 0
        loss = 0

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
            print('Hold')
            pass  # do nothing

        done = True if t == l - 1 else False
        if done:
            print("--------------------------------")
            print("Total Profit: " + format_price(total_profit))
            print("--------------------------------")
            exit()

        agent.memory.append((state, actions, reward, next_state, done))
        state = next_state

        if len(agent.memory) > agent.batch_size:
            loss += agent.experience_replay(agent.batch_size, e, t, loss)
            # print("Episode", e, "Step", t, "Action", action, "Reward", reward, "Loss", loss)

    if e % 10 == 0:
        agent.actor.save("saved_models/DDPG_ep" + str(e))
        print('model saved')
