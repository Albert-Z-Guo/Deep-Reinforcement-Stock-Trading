from collections import deque

import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential, clone_model
from keras.models import load_model
from keras.layers import Dense, Concatenate, Input, add
from keras.initializers import VarianceScaling
from keras import backend as K
from keras.optimizers import Adam


class ActorNetwork:
    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate, is_eval=False, model_name=""):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        K.set_session(sess)
        if is_eval == True:
            self.model = load_model("saved_models/" + model_name)
        else:
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
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        buy = Dense(1, activation='tanh')(h1)
        hold = Dense(1, activation='sigmoid')(h1)
        sell = Dense(1, activation='sigmoid')(h1)
        A = Concatenate()([buy, hold, sell])
        model = Model(inputs=S, outputs=A)
        return model, model.trainable_weights, S


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class CriticNetwork:
    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate):
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
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = add([h1, a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        Q = Dense(action_dim, activation='linear')(h3)
        model = Model(inputs=[S, A], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model, A, S


class Agent:
    def __init__(self, state_dim, is_eval=False, model_name=""):
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.inventory = []
        self.balance = 1000000
        self.gamma = 0.95 # discount factor
        self.epsilon = 1.0 # initial exploration rate
        self.epsilon_min = 0.1 # minimum exploration rate
        self.epsilon_decay = 0.99995
        self.is_eval = is_eval
        tau = 0.001  # Target Network Hyper Parameter
        LRA = 0.0001  # learning rate for Actor Network
        LRC = 0.001  # learning rate for Critic Network
        # Tensorflow GPU configuration
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        self.actor = ActorNetwork(sess, state_dim, self.action_dim, self.batch_size, tau, LRA, is_eval, model_name)
        self.critic = CriticNetwork(sess, state_dim, self.action_dim, self.batch_size, tau, LRC)

    def remember(self, state, action, reward, next_state, done):
    	self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            exploration_noise = np.random.normal(loc=0, scale=0.5, size=self.action_dim)
            return np.argmax(self.actor.model.predict(state)[0] + exploration_noise)
        return np.argmax(self.actor.model.predict(state)[0])

    def experience_replay(self, batch_size, e, t, loss):
        # retrieve recent batch_size long memory from deque
        mini_batch = []
        memory_len = len(self.memory)
        for i in range(memory_len - batch_size + 1, memory_len):
            mini_batch.append(self.memory[i])

        y_batch = []
        state_batch = []
        actions_batch = []
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                target_q_values = self.critic.target_model.predict([next_state, self.actor.target_model.predict(next_state)])
                y = reward + self.gamma * target_q_values
            else:
                y = reward * np.ones((1, self.action_dim))
            y_batch.append(y)
            state_batch.append(state)
            actions_batch.append(actions)

        y_batch = np.vstack(y_batch)
        state_batch = np.vstack(state_batch)
        actions_batch = np.vstack(actions_batch)

        # update networks
        loss += self.critic.model.train_on_batch([state_batch, actions_batch], y_batch)
        grads = self.critic.gradients(state_batch, self.actor.model.predict(state_batch))
        self.actor.train(state_batch, grads)
        self.actor.target_train()
        self.critic.target_train()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss
