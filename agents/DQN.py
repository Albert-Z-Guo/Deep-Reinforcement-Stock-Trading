import random
from collections import deque

import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self, state_size, balance=50000, is_eval=False, model_name=""):
        self.model_type = 'DQN'
        self.state_size = state_size  # normalized previous days
        self.action_space = 3  # hold, buy, sell
        self.memory = deque(maxlen=1000)
        self.initial_portfolio_value = balance
        self.balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = []
        self.buy_dates = []
        self.sell_dates = []

        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995
        self.is_eval = is_eval
        self.model = load_model("saved_models/" + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self, batch_size):
        # retrieve recent batch_size long memory
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            target_future = self.model.predict(state)
            target_future[0][action] = target
            self.model.fit(state, target_future, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
