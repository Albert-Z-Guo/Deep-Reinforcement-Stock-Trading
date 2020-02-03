import argparse
import importlib
import logging
import sys
import time

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Hold', 2: 'Sell'}

# select learning model
model = importlib.import_module('agents.{}'.format(model_name))
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)

def hold(actions):
    # encourage selling for profit and liquidity
    next_probable_action = np.argsort(actions)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell(t)
            actions[next_probable_action] = 1 # reset this action's value to the highest
            return 'Hold', actions

def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)

# configure logger
logger = logging.getLogger()
handler = logging.FileHandler('logs/{}_training_{}.log'.format(model_name, stock_name), mode='w')
handler.setFormatter(logging.Formatter(fmt='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info('Trading Object:           {}'.format(stock_name))
logger.info('Trading Period:           {} days'.format(trading_period))
logger.info('Window Size:              {} days'.format(window_size))
logger.info('Training Episode:         {}'.format(num_episode))
logger.info('Model Name:               {}'.format(model_name))
logger.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

start_time = time.time()
for e in range(1, num_episode + 1):
    logger.info('\nEpisode: {}/{}'.format(e, num_episode))

    agent.reset() # reset to initial balance and hyperparameters
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            logger.info('\n-------------------Period: {}/{}-------------------'.format(t, trading_period))

        reward = 0
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
        previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)
        
        # execute position
        logger.info('Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1], actions[2]))
        if action != np.argmax(actions): logger.info("\t\t'{}' is an exploration.".format(action_dict[action]))
        if action == 0: # hold
            execution_result = hold(actions)
        if action == 1: # buy
            execution_result = buy(t)      
        if action == 2: # sell
            execution_result = sell(t)        
        
        # check execution result
        if execution_result is None:
            reward -= daily_treasury_bond_return_rate() * agent.balance  # missing opportunity
        else:
            if isinstance(execution_result, tuple): # if execution_result is 'Hold'
                actions = execution_result[1]
                execution_result = execution_result[0]
            logger.info(execution_result)                

        # calculate reward
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        reward += unrealized_profit

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = True if t == trading_period else False
        agent.remember(state, actions, reward, next_state, done)

        # update state
        state = next_state

        # experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            loss = agent.experience_replay()
            logger.info('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e, loss, action_dict[action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logger)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 5 == 0:
        if model_name == 'DQN':
            agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
        elif model_name == 'DDPG':
            agent.actor.model.save_weights('saved_models/DDPG_actor_ep' + str(e) + '.h5')
        logger.info('model saved')

logger.info('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
