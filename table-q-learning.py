import gym
import numpy as np

learning_rate = 0.8
discount_rate = 0.95
num_episodes = 10000
max_episode_step = 3000

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

rewards_from_all_episodes = []
for episode in range(num_episodes):
    episode_reward = 0
    observation = env.reset()
    for step in range(max_episode_step):
        action = np.argmax(Q[observation, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))
        new_observation, reward, done, _ = env.step(action)
        Q[observation, action] = Q[observation, action] + learning_rate * \
                                 (reward + discount_rate * np.max(Q[new_observation, :]) - Q[observation, action])
        episode_reward += reward
        observation = new_observation
        if done:
            break
    rewards_from_all_episodes.append(episode_reward)


print("Score over time: {}".format(sum(rewards_from_all_episodes) / num_episodes))
