import gym
import numpy as np
import tensorflow as tf
import random
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.initializers import *

env = gym.make('FrozenLake-v0')

learning_rate = 0.0001
discount_rate = 0.99
random_action_chance = 0.1
num_episodes = 100000
max_episode_step = 100
log_interval = 100


model = tf.keras.Sequential()
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
model.add(Dense(4, kernel_initializer='uniform', activation='softmax'))
model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
              loss='mean_squared_error')

rewards_from_episodes = []

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0
    if episode % log_interval == 0 and episode > 0:
        print('Episode: {}, Reward: {}'.format(episode, sum(
            rewards_from_episodes[episode - log_interval: episode]) / log_interval))

    for step in range(max_episode_step):
        # Select action
        targetQ = model.predict(np.identity(16)[observation:observation + 1], batch_size=1)
        # for layer in model.layers:
        #     print('layer: {}, weight: {}'.format(layer, layer.get_weights()))
        action = np.argmax(targetQ)

        if random.random() < random_action_chance:
            action = env.action_space.sample()

        new_observation, reward, done, _ = env.step(action)
        Qnew = model.predict(np.identity(16)[new_observation:new_observation + 1], batch_size=1)
        maxQvalue = np.max(Qnew)
        targetQ[0, action] = reward + discount_rate * maxQvalue

        # Train network using target and predicted Q values
        model.fit(np.identity(16)[observation:observation + 1], targetQ, epochs=1, batch_size=1, verbose=0)

        episode_reward += reward
        observation = new_observation
        if done:
            random_action_chance = 1. / ((episode / 50) + 10)
            break
    rewards_from_episodes.append(episode_reward)

print("Mean of all episodes: {}%".format(sum(rewards_from_episodes) / num_episodes))
