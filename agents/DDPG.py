import random
from collections import deque

import numpy as np
# np.random.seed(1) # for reproducible Keras operations
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Concatenate
from keras import backend as K
from keras.optimizers import Adam

from utils import OUNoise


HIDDEN1_UNITS = 24
HIDDEN2_UNITS = 48


class ActorNetwork:
    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        if is_eval == True:
            self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
            self.model.load_weights("saved_models/" + model_name)
        else:
            K.set_session(sess)
            self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
            self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
            self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
            self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
            self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(self.params_grad, self.weights))
            self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={self.state: states, self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(states)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        hold = Dense(1, activation='sigmoid')(h1)
        buy = Dense(1, activation='sigmoid')(h1)
        sell = Dense(1, activation='sigmoid')(h1)
        actions = Concatenate()([hold, buy, sell])
        model = Model(inputs=states, outputs=actions)
        return model, model.trainable_weights, states


class CriticNetwork:
    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
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
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        actions = Input(shape=[action_dim])
        h0 = Concatenate()([states, actions])
        h1 = Dense(HIDDEN1_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        Q = Dense(action_dim, activation='relu')(h2)
        model = Model(inputs=[states, actions], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=1e-6))
        return model, actions, states


class Agent:
    def __init__(self, state_dim, initial_funding=10000, is_eval=False, model_name=""):
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.batch_size = 60
        self.balance = initial_funding
        self.inventory = []

        self.gamma = 0.95 # discount factor
        self.is_eval = is_eval
        self.noise = OUNoise(self.action_dim)
        tau = 0.001  # Target Network Hyper Parameter
        LRA = 0.001  # learning rate for Actor Network
        LRC = 0.001  # learning rate for Critic Network

        # Tensorflow GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.actor = ActorNetwork(sess, state_dim, self.action_dim, self.batch_size, tau, LRA, is_eval, model_name)
        self.critic = CriticNetwork(sess, state_dim, self.action_dim, self.batch_size, tau, LRC)

    def remember(self, state, actions, reward, next_state, done):
    	self.memory.append((state, actions, reward, next_state, done))

    def act(self, state, t):
        actions = self.actor.model.predict(state)[0]
        if not self.is_eval:
            return self.noise.get_actions(actions, t)
        return actions

    def experience_replay(self, batch_size, e, t):
        # retrieve random batch_size long memory from deque
        mini_batch = random.sample(self.memory, batch_size)

        y_batch = []
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                target_q_values = self.critic.target_model.predict([next_state, self.actor.target_model.predict(next_state)])
                y = reward + self.gamma * target_q_values
            else:
                y = reward * np.ones((1, self.action_dim))
            y_batch.append(y)

        y_batch = np.vstack(y_batch)
        state_batch = np.vstack([tup[0] for tup in mini_batch])
        actions_batch = np.vstack([tup[1] for tup in mini_batch])

        # update networks
        loss = self.critic.model.train_on_batch([state_batch, actions_batch], y_batch)
        grads = self.critic.gradients(state_batch, self.actor.model.predict(state_batch))
        self.actor.train(state_batch, grads)
        self.actor.target_train()
        self.critic.target_train()
        return loss
