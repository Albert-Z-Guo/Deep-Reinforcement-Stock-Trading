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

Currently, the state is defined as the normalized adjacent daily stock price differences for `n` days plus  `[stock_price, balance, num_holding]`.

In the future, we plan to add other state-of-the-art deep reinforcement learning algorithms, such as Proximal Policy Optimization (PPO), to the framework and increase the complexity to the state in each algorithm by constructing more complex price tensors etc. with a wider range of deep learning approaches, such as convolutional neural networks or attention mechanism.

### Getting Started
To install all libraries/dependencies used in this project, run
```bash
pip3 install -r requirement.txt
```

To train a DDPG agent or a DQN agent (with window size included), e.g. over S&P 500 from 2010 to 2015, run
```bash
python3 train_DDPG.py ^GSPC_2010-2015 [window size] [epoch number]
python3 train_DQN.py ^GSPC_2010-2015 [window size] [epoch number]
```

To evaluate a DDPG or DQN agent, run
```bash
python3 evaluate_DDPG.py [stock symbol] [model name]
python3 evaluate_DQN.py [stock symbol] [model name]
```

where stock symbols can be referred in `data` folder and model names can be referred in `saved_models`.

### Example Results
![alt_text](./visualizations/DQN_^GSPC_2014.png)

### References:
- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)
- [Adversarial Deep Reinforcement Learning in Portfolio Management](https://arxiv.org/abs/1808.09940)
- [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059)
