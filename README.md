# Deep-Reinforcement-Stock-Trading

This project intends to leverage deep reinforcement learning in portfolio management.

The light-weight Deep Q-network (DQN) is inspired by [Q-Trader](https://github.com/edwardhdlu/q-trader) and the Deep Deterministic Policy Gradient (DDPG) agent structure is adapted from [DDPG-Keras-Torcs](https://github.com/yanpanlau/DDPG-Keras-Torcs) with deep integration. All evaluation metrics and visualizations are built from scratch.

Key assumptions and limitations of the current frameworks:
- trading has no impact on the market
- only single stock type is supported
- only 3 actions: buy, hold, sell
- the agent performs only 1 action for portfolio reallocation at the end of each trade day
- all reallocations can be finished at the closing prices
- no missing data in price history
- no transaction cost

Key challenges of the current framework:
- building a reliable reward mechanism
- ensuring the framework is scalable and extensible

Currently, the state is defined as the normalized adjacent daily stock price differences for `n` days plus  `[stock_price, balance, num_holding]`.

In the future, we plan to add other state-of-the-art deep reinforcement learning algorithms, such as Proximal Policy Optimization (PPO), to the framework and increase the complexity to the state in each algorithm by constructing more complex price tensors etc. with a wider range of deep learning approaches, such as convolutional neural networks or attention mechanism. In addition, we plan to integrate better pipelines for high quality data source, e.g. from vendors like [Quandl](https://www.quandl.com/); and backtesting, e.g. [zipline](https://github.com/quantopian/zipline).

### Getting Started
To install all libraries/dependencies used in this project, run
```bash
pip3 install -r requirement.txt
```

To train a DDPG agent or a DQN agent (with window size included), e.g. over S&P 500 from 2010 to 2015, run
```bash
python3 train.py --model_name=model_name --stock_name=stock_name
```

- `model_name`      is the model to use: either `DQN` or `DDPG`; default is `DQN`
- `stock_name`      is the stock used to train the model; default is `^GSPC_2010-2015`, which is S&P 500 from 1/1/2010 to 12/31/2015
- `window_size`     is the span (days) of observation; default is `10`
- `num_episode`     is the number of episodes used for training; default is `10`
- `initial_funding` is the initial funding of the portfolio; default is `50000`

To evaluate a DDPG or DQN agent, run
```bash
python3 evaluate.py --model_name=model_name --model_to_load=model_to_load --stock_name=stock_name
```

- `model_name`      is the model to use: either `DQN` or `DDPG`; default is `DQN`
- `model_to_laod`   is the model to load; default is `DQN_ep1.h5`
- `stock_name`   is the stock used to evaluate the model; default is `^GSPC_2018`, which is S&P 500 from 1/1/2018 to 12/31/2018
- `initial_funding` is the initial funding of the portfolio; default is `50000`

where `stock_name` can be referred in `data` directory and `model_to_laod` can be referred in `saved_models` directory.

### Example Results
![alt_text](./visualizations/DQN_^GSPC_2014.png)

### References:
- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)
- [Adversarial Deep Reinforcement Learning in Portfolio Management](https://arxiv.org/abs/1808.09940)
- [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059)
