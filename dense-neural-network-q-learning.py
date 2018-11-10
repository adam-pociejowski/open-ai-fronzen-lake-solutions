import gym
import numpy as np
import tensorflow as tf

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()
inputs = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Q = tf.matmul(inputs, W)
predict = tf.argmax(Q, 1)

Qnext = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Qnext - Q))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = optimizer.minimize(loss)

init = tf.initialize_all_variables()

discount_rate = 0.99
random_action_chance = 0.1
num_episodes = 10000
max_episode_step = 100
log_interval = 100
rewards_from_episodes = []
with tf.Session() as sess:
    sess.run(init)
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        if episode % log_interval == 0 and episode > 0:
            print('Episode: {}, Reward: {}'.format(episode, sum(rewards_from_episodes[episode - log_interval: episode]) / log_interval))

        W1 = []
        for step in range(max_episode_step):
            # Select action
            action, targetQ = sess.run([predict, Q], feed_dict={inputs: np.identity(16)[observation:observation + 1]})
            if np.random.rand(1) < random_action_chance:
                action[0] = env.action_space.sample()

            new_observation, reward, done, _ = env.step(action[0])
            Qnew = sess.run(Q, feed_dict={inputs: np.identity(16)[new_observation:new_observation + 1]})
            maxQvalue = np.max(Qnew)
            targetQ[0, action[0]] = reward + discount_rate * maxQvalue
            # Train network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs: np.identity(16)[observation:observation + 1],
                                                          Qnext: targetQ})
            episode_reward += reward
            observation = new_observation
            if done:
                random_action_chance = 1. / ((episode / 50) + 10)
                break
        rewards_from_episodes.append(episode_reward)

print("Percent of succesful episodes: {}%".format(sum(rewards_from_episodes) / num_episodes))
