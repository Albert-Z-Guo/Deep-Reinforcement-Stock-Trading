import sys
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential, clone_model
from keras.models import load_model
from keras.layers import Dense, Concatenate, Input, add
from keras.initializers import VarianceScaling
from keras import backend as K
from keras.optimizers import Adam

from utils import *

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        K.set_session(sess)
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.params_grad, self.weights))
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={self.state: states, self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        buy = Dense(1,activation='tanh',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)
        hold = Dense(1,activation='sigmoid',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)
        sell = Dense(1,activation='sigmoid',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)
        A = Concatenate()([buy, hold, sell])
        model = Model(inputs=S, outputs=A)
        return model, model.trainable_weights, S


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        K.set_session(sess)
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={self.state: states, self.action: actions})[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = add([h1,a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        Q = Dense(action_dim, activation='linear')(h3)
        model = Model(inputs=[S,A], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(lr=self.LEARNING_RATE))
        return model, A, S


class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size  # normalized previous days
		self.action_space = 3  # hold, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_space)
		options = self.model.predict(state)
		return np.argmax(options[0])


if len(sys.argv) != 4:
	print("Usage: python train.py [stock] [window] [episodes]")
	exit()
stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


stock_prices = stock_close_prices(stock_name)
l = len(stock_prices) - 1

BATCH_SIZE = 32
state_dim = window_size + 2
action_dim = 3  # hold, buy, sell
memory = deque(maxlen=1000)
inventory = []
GAMMA = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
is_eval = False
TAU = 0.001  # Target Network Hyper Parameter
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic

# Tensorflow GPU configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)


for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    total_profit = 0
	# skip random process for action exploration
    state = generate_ddpg_state(stock_prices, 0, window_size + 1, total_profit, len(inventory))

    for t in range(l):
        actions = actor.model.predict(state)
        action = np.argmax(actor.model.predict(state)[0])
        next_state = generate_ddpg_state(stock_prices, t + 1, window_size + 1, total_profit, len(inventory))
        reward = 0
        loss = 0

        # buy
        if action == 1:
            inventory.append(stock_prices[t])
            print("Buy: " + format_price(stock_prices[t]))
        # sell
        elif action == 2 and len(inventory) > 0:
            bought_price = inventory.pop(0)
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

        memory.append((state, actions, reward, next_state, done))
        state = next_state

        if len(memory) > BATCH_SIZE:
            # retrieve recent batch_size long memory
            mini_batch = []
            memory_len = len(memory)
            for i in range(memory_len - BATCH_SIZE + 1, memory_len):
                mini_batch.append(memory[i])

            y_batch = []
            state_batch = []
            actions_batch = []
            for state, actions, reward, next_state, done in mini_batch:
                if not done:
                    target_q_values = critic.target_model.predict([next_state, actor.target_model.predict(next_state)])
                    y = reward + GAMMA * target_q_values
                else:
                    y = reward * np.ones((1, action_dim))
                y_batch.append(y)
                state_batch.append(state)
                actions_batch.append(actions)

            y_batch = np.vstack(y_batch)
            state_batch = np.vstack(state_batch)
            actions_batch = np.vstack(actions_batch)

            loss += critic.model.train_on_batch([state_batch, actions_batch], y_batch)
            grads = critic.gradients(state_batch, actor.model.predict(state_batch))
            actor.train(state_batch, grads)
            actor.target_train()
            critic.target_train()

            # print("Episode", e, "Step", t, "Action", action, "Reward", reward, "Loss", loss)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
