import random
from collections import deque

import numpy as np
# np.random.seed(3)  # for reproducible Keras operations
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam


# Tensorflow GPU configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()


HIDDEN1_UNITS = 24
HIDDEN2_UNITS = 48
HIDDEN3_UNITS = 24


class ActorNetwork:
    def __init__(self, sess, state_size, action_dim, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        if is_eval == True:
            self.model, self.weights, self.state = self.create_actor_network(state_size, action_dim)
            self.model.load_weights("saved_models/" + model_name)
        else:
            self.model, self.weights, self.state = self.create_actor_network(state_size, action_dim)
            self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_dim)
            self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, action_dim])
            self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
            self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.params_grad, self.weights))

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={self.state: states, self.action_gradient: action_grads})

    def train_target(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(states)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='relu')(h1)
        actions = Dense(self.action_dim, activation='softmax')(h2)
        model = Model(inputs=states, outputs=actions)
        return model, model.trainable_weights, states


class CriticNetwork:
    def __init__(self, sess, state_size, action_dim, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.model, self.action, self.state = self.create_critic_network(state_size, action_dim)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_dim)
        self.action_grads = tf.gradients(self.model.output, self.action)

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={self.state: states, self.action: actions})[0]

    def train_target(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        actions = Input(shape=[action_dim])
        h0 = Concatenate()([states, actions])
        h1 = Dense(HIDDEN1_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu')(h2)
        Q = Dense(action_dim, activation='relu')(h3)
        model = Model(inputs=[states, actions], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=1e-6))
        return model, actions, states


# reference: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_actions(self, actions, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(actions + ou_state, 0, 1)


class Agent:
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        self.model_type = 'DDPG'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 90
        self.initial_portfolio_value = balance
        self.balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [balance]
        self.buy_dates = []
        self.sell_dates = []

        self.gamma = 0.95 # discount factor
        self.is_eval = is_eval
        self.noise = OUNoise(self.action_dim)
        tau = 0.001  # Target network hyperparameter
        learning_rate_actor = 0.001  # learning rate for Actor network
        learning_rate_critic = 0.001  # learning rate for Critic network

        self.actor = ActorNetwork(sess, state_dim, self.action_dim, tau, learning_rate_actor, is_eval, model_name)
        self.critic = CriticNetwork(sess, state_dim, self.action_dim, tau, learning_rate_critic)
        sess.run(tf.compat.v1.global_variables_initializer())

        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/DDPG', update_freq=90)
        self.tensorboard.set_model(self.critic.model)

    def reset(self, balance):
        self.balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [balance]
        self.noise.reset()

    def remember(self, state, actions, reward, next_state, done):
    	self.memory.append((state, actions, reward, next_state, done))

    def act(self, state, t):
        actions = self.actor.model.predict(state)[0]
        if not self.is_eval:
            return self.noise.get_actions(actions, t)
        return actions

    def experience_replay(self):
        # sample random buffer_size long memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        y_batch = []
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = self.critic.target_model.predict([next_state, self.actor.target_model.predict(next_state)])
                y = reward + self.gamma * Q_target_value
            else:
                y = reward * np.ones((1, self.action_dim))
            y_batch.append(y)

        y_batch = np.vstack(y_batch)
        state_batch = np.vstack([tup[0] for tup in mini_batch])
        actions_batch = np.vstack([tup[1] for tup in mini_batch])
        
        # update critic by minimizing the loss
        loss = self.critic.model.train_on_batch([state_batch, actions_batch], y_batch)

        # update actor using the sampled policy gradients
        grads = self.critic.gradients(state_batch, self.actor.model.predict(state_batch))
        self.actor.train(state_batch, grads)
        
        # update target networks
        self.actor.train_target()
        self.critic.train_target()
        return loss
