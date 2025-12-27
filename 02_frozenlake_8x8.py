#!/usr/bin/env python
# coding: utf-8

# # FrozenLake [8x8 | no random map | on slip]

# In[29]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch device:", device)


# ### Model

# In[22]:


class FrozenLake8x8V0(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(FrozenLake8x8V0, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)


# ## Replay Buffer

# In[23]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ## Agent

# In[24]:


class DQNAgent:
    def __init__(self, modelClass, state_size, action_size, lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=64, target_update=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Q-network and target network
        self.q_net = modelClass(state_size, action_size).to(device)
        self.target_net = modelClass(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Sync target network
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayBuffer(buffer_size)
        self.step_count = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # current q-values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # next q-values frm target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ## Training Loop

# In[30]:


def one_hot_state(s, size):
    vec = np.zeros(size)
    vec[s] = 1.0
    return vec


def train_dqn(modelClass, env_name="FrozenLake8x8-v1", is_slippery=False, episodes=2000, max_steps=100):
    env = gym.make(env_name, is_slippery=is_slippery, render_mode=None)
    state_size = env.observation_space.n
    action_size = env.action_space.n  # 4 (left, down, right, up)

    agent = DQNAgent(modelClass=modelClass, state_size=state_size, action_size=action_size)

    scores = deque(maxlen=100)  # for moving average
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99999

    epsilon = epsilon_start

    # Track metrics for plotting
    avg_rewards = []
    epsilons = []

    print("Starting training...")
    for episode in range(episodes):
        state, _ = env.reset()
        state = one_hot_state(state, state_size)
        total_reward = 0

        for t in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # FrozenLake gives sparse reward: 1 only if goal reached
            reward = reward - 0.01  # optional shaping to encourage speed

            next_state = one_hot_state(next_state, state_size)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Log every episode for smooth plot
        avg_score = np.mean(scores)
        avg_rewards.append(avg_score)
        epsilons.append(epsilon)

        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Avg Reward (last 100): {avg_score:.3f}, Epsilon: {epsilon:.3f}")

    env.close()
    return agent, env, (avg_rewards, epsilons)


def evaluate_agent(agent, env, episodes=10, max_steps=100):
    state_size = env.observation_space.n

    success = 0
    for _ in range(episodes):
        state, _ = env.reset()
        state = one_hot_state(state, state_size)
        for _ in range(max_steps):
            action = agent.act(state, epsilon=0.0)  # greedy
            state, reward, terminated, truncated, _ = env.step(action)
            state = one_hot_state(state, state_size)
            if terminated or truncated:
                if reward == 1.0:
                    success += 1
                break
    print(f"\nSuccess rate over {episodes} episodes: {success}/{episodes} ({100 * success / episodes:.1f}%)")


# In[31]:


# todo: continuable & interruptible training loop
agent, env, (avg_rewards, epsilons) = train_dqn(modelClass=FrozenLake8x8V0, is_slippery=False, episodes=200)

# Plotting
print(f"\\nModel: {FrozenLake8x8V0.__name__}", agent.q_net)
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Avg Reward (last 100)', color=color)
ax1.plot(avg_rewards, color=color, label='Avg Reward')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(epsilons, color=color, label='Epsilon', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Progress')
fig.tight_layout()
plt.show()


# OUT[28]:
"""
Starting training...
Episode 0, Avg Reward (last 100): -0.280, Epsilon: 1.000
Episode 100, Avg Reward (last 100): -0.311, Epsilon: 0.999
Episode 200, Avg Reward (last 100): -0.321, Epsilon: 0.998
Episode 300, Avg Reward (last 100): -0.278, Epsilon: 0.997
Episode 400, Avg Reward (last 100): -0.319, Epsilon: 0.996
Episode 500, Avg Reward (last 100): -0.343, Epsilon: 0.995
Episode 600, Avg Reward (last 100): -0.362, Epsilon: 0.994
Episode 700, Avg Reward (last 100): -0.260, Epsilon: 0.993
Episode 800, Avg Reward (last 100): -0.310, Epsilon: 0.992
Episode 900, Avg Reward (last 100): -0.296, Epsilon: 0.991
Episode 1000, Avg Reward (last 100): -0.268, Epsilon: 0.990
Episode 1100, Avg Reward (last 100): -0.280, Epsilon: 0.989
Episode 1200, Avg Reward (last 100): -0.337, Epsilon: 0.988
Episode 1300, Avg Reward (last 100): -0.332, Epsilon: 0.987
Episode 1400, Avg Reward (last 100): -0.302, Epsilon: 0.986
Episode 1500, Avg Reward (last 100): -0.321, Epsilon: 0.985
Episode 1600, Avg Reward (last 100): -0.316, Epsilon: 0.984
Episode 1700, Avg Reward (last 100): -0.321, Epsilon: 0.983
Episode 1800, Avg Reward (last 100): -0.312, Epsilon: 0.982
Episode 1900, Avg Reward (last 100): -0.345, Epsilon: 0.981
Episode 2000, Avg Reward (last 100): -0.337, Epsilon: 0.980
Episode 2100, Avg Reward (last 100): -0.282, Epsilon: 0.979
Episode 2200, Avg Reward (last 100): -0.317, Epsilon: 0.978
Episode 2300, Avg Reward (last 100): -0.297, Epsilon: 0.977
Episode 2400, Avg Reward (last 100): -0.272, Epsilon: 0.976
Episode 2500, Avg Reward (last 100): -0.269, Epsilon: 0.975
Episode 2600, Avg Reward (last 100): -0.276, Epsilon: 0.974
Episode 2700, Avg Reward (last 100): -0.274, Epsilon: 0.973
Episode 2800, Avg Reward (last 100): -0.323, Epsilon: 0.972
Episode 2900, Avg Reward (last 100): -0.344, Epsilon: 0.971
Episode 3000, Avg Reward (last 100): -0.246, Epsilon: 0.970
Episode 3100, Avg Reward (last 100): -0.319, Epsilon: 0.969
Episode 3200, Avg Reward (last 100): -0.301, Epsilon: 0.968
Episode 3300, Avg Reward (last 100): -0.320, Epsilon: 0.968
Episode 3400, Avg Reward (last 100): -0.287, Epsilon: 0.967
Episode 3500, Avg Reward (last 100): -0.277, Epsilon: 0.966
Episode 3600, Avg Reward (last 100): -0.257, Epsilon: 0.965
Episode 3700, Avg Reward (last 100): -0.258, Epsilon: 0.964
Episode 3800, Avg Reward (last 100): -0.290, Epsilon: 0.963
Episode 3900, Avg Reward (last 100): -0.276, Epsilon: 0.962
Episode 4000, Avg Reward (last 100): -0.294, Epsilon: 0.961
Episode 4100, Avg Reward (last 100): -0.329, Epsilon: 0.960
Episode 4200, Avg Reward (last 100): -0.282, Epsilon: 0.959
Episode 4300, Avg Reward (last 100): -0.304, Epsilon: 0.958
Episode 4400, Avg Reward (last 100): -0.329, Epsilon: 0.957
Episode 4500, Avg Reward (last 100): -0.255, Epsilon: 0.956
Episode 4600, Avg Reward (last 100): -0.305, Epsilon: 0.955
Episode 4700, Avg Reward (last 100): -0.316, Epsilon: 0.954
Episode 4800, Avg Reward (last 100): -0.279, Epsilon: 0.953
Episode 4900, Avg Reward (last 100): -0.320, Epsilon: 0.952
Episode 5000, Avg Reward (last 100): -0.248, Epsilon: 0.951
Episode 5100, Avg Reward (last 100): -0.284, Epsilon: 0.950
Episode 5200, Avg Reward (last 100): -0.299, Epsilon: 0.949
Episode 5300, Avg Reward (last 100): -0.316, Epsilon: 0.948
Episode 5400, Avg Reward (last 100): -0.274, Epsilon: 0.947
Episode 5500, Avg Reward (last 100): -0.296, Epsilon: 0.946
Episode 5600, Avg Reward (last 100): -0.291, Epsilon: 0.946
Episode 5700, Avg Reward (last 100): -0.303, Epsilon: 0.945
Episode 5800, Avg Reward (last 100): -0.271, Epsilon: 0.944
Episode 5900, Avg Reward (last 100): -0.280, Epsilon: 0.943
Episode 6000, Avg Reward (last 100): -0.286, Epsilon: 0.942
Episode 6100, Avg Reward (last 100): -0.276, Epsilon: 0.941
Episode 6200, Avg Reward (last 100): -0.270, Epsilon: 0.940
Episode 6300, Avg Reward (last 100): -0.309, Epsilon: 0.939
Episode 6400, Avg Reward (last 100): -0.311, Epsilon: 0.938
Episode 6500, Avg Reward (last 100): -0.263, Epsilon: 0.937
Episode 6600, Avg Reward (last 100): -0.304, Epsilon: 0.936
Episode 6700, Avg Reward (last 100): -0.254, Epsilon: 0.935
Episode 6800, Avg Reward (last 100): -0.290, Epsilon: 0.934
Episode 6900, Avg Reward (last 100): -0.290, Epsilon: 0.933
Episode 7000, Avg Reward (last 100): -0.273, Epsilon: 0.932
Episode 7100, Avg Reward (last 100): -0.299, Epsilon: 0.931
Episode 7200, Avg Reward (last 100): -0.255, Epsilon: 0.931
Episode 7300, Avg Reward (last 100): -0.276, Epsilon: 0.930
Episode 7400, Avg Reward (last 100): -0.290, Epsilon: 0.929
Episode 7500, Avg Reward (last 100): -0.256, Epsilon: 0.928
Episode 7600, Avg Reward (last 100): -0.255, Epsilon: 0.927
Episode 7700, Avg Reward (last 100): -0.284, Epsilon: 0.926
Episode 7800, Avg Reward (last 100): -0.270, Epsilon: 0.925
Episode 7900, Avg Reward (last 100): -0.262, Epsilon: 0.924
Episode 8000, Avg Reward (last 100): -0.268, Epsilon: 0.923
Episode 8100, Avg Reward (last 100): -0.289, Epsilon: 0.922
Episode 8200, Avg Reward (last 100): -0.289, Epsilon: 0.921
Episode 8300, Avg Reward (last 100): -0.248, Epsilon: 0.920
Episode 8400, Avg Reward (last 100): -0.273, Epsilon: 0.919
Episode 8500, Avg Reward (last 100): -0.230, Epsilon: 0.919
Episode 8600, Avg Reward (last 100): -0.302, Epsilon: 0.918
Episode 8700, Avg Reward (last 100): -0.261, Epsilon: 0.917
Episode 8800, Avg Reward (last 100): -0.271, Epsilon: 0.916
Episode 8900, Avg Reward (last 100): -0.272, Epsilon: 0.915
Episode 9000, Avg Reward (last 100): -0.263, Epsilon: 0.914
Episode 9100, Avg Reward (last 100): -0.280, Epsilon: 0.913
Episode 9200, Avg Reward (last 100): -0.295, Epsilon: 0.912
Episode 9300, Avg Reward (last 100): -0.257, Epsilon: 0.911
Episode 9400, Avg Reward (last 100): -0.264, Epsilon: 0.910
Episode 9500, Avg Reward (last 100): -0.269, Epsilon: 0.909
Episode 9600, Avg Reward (last 100): -0.267, Epsilon: 0.908
Episode 9700, Avg Reward (last 100): -0.260, Epsilon: 0.908
Episode 9800, Avg Reward (last 100): -0.262, Epsilon: 0.907
Episode 9900, Avg Reward (last 100): -0.288, Epsilon: 0.906
Episode 10000, Avg Reward (last 100): -0.213, Epsilon: 0.905
Episode 10100, Avg Reward (last 100): -0.253, Epsilon: 0.904
Episode 10200, Avg Reward (last 100): -0.272, Epsilon: 0.903
Episode 10300, Avg Reward (last 100): -0.277, Epsilon: 0.902
Episode 10400, Avg Reward (last 100): -0.266, Epsilon: 0.901
Episode 10500, Avg Reward (last 100): -0.245, Epsilon: 0.900
Episode 10600, Avg Reward (last 100): -0.275, Epsilon: 0.899
Episode 10700, Avg Reward (last 100): -0.291, Epsilon: 0.899
Episode 10800, Avg Reward (last 100): -0.287, Epsilon: 0.898
Episode 10900, Avg Reward (last 100): -0.268, Epsilon: 0.897
Episode 11000, Avg Reward (last 100): -0.257, Epsilon: 0.896
Episode 11100, Avg Reward (last 100): -0.267, Epsilon: 0.895
Episode 11200, Avg Reward (last 100): -0.260, Epsilon: 0.894
Episode 11300, Avg Reward (last 100): -0.266, Epsilon: 0.893
Episode 11400, Avg Reward (last 100): -0.294, Epsilon: 0.892
Episode 11500, Avg Reward (last 100): -0.235, Epsilon: 0.891
Episode 11600, Avg Reward (last 100): -0.291, Epsilon: 0.890
Episode 11700, Avg Reward (last 100): -0.267, Epsilon: 0.890
Episode 11800, Avg Reward (last 100): -0.251, Epsilon: 0.889
Episode 11900, Avg Reward (last 100): -0.256, Epsilon: 0.888
Episode 12000, Avg Reward (last 100): -0.258, Epsilon: 0.887
Episode 12100, Avg Reward (last 100): -0.261, Epsilon: 0.886
Episode 12200, Avg Reward (last 100): -0.288, Epsilon: 0.885
Episode 12300, Avg Reward (last 100): -0.258, Epsilon: 0.884
Episode 12400, Avg Reward (last 100): -0.287, Epsilon: 0.883
Episode 12500, Avg Reward (last 100): -0.256, Epsilon: 0.882
Episode 12600, Avg Reward (last 100): -0.242, Epsilon: 0.882
Episode 12700, Avg Reward (last 100): -0.260, Epsilon: 0.881
Episode 12800, Avg Reward (last 100): -0.282, Epsilon: 0.880
Episode 12900, Avg Reward (last 100): -0.251, Epsilon: 0.879
Episode 13000, Avg Reward (last 100): -0.216, Epsilon: 0.878
Episode 13100, Avg Reward (last 100): -0.254, Epsilon: 0.877
Episode 13200, Avg Reward (last 100): -0.231, Epsilon: 0.876
Episode 13300, Avg Reward (last 100): -0.253, Epsilon: 0.875
Episode 13400, Avg Reward (last 100): -0.251, Epsilon: 0.875
Episode 13500, Avg Reward (last 100): -0.232, Epsilon: 0.874
Episode 13600, Avg Reward (last 100): -0.262, Epsilon: 0.873
Episode 13700, Avg Reward (last 100): -0.254, Epsilon: 0.872
Episode 13800, Avg Reward (last 100): -0.230, Epsilon: 0.871
Episode 13900, Avg Reward (last 100): -0.242, Epsilon: 0.870
Episode 14000, Avg Reward (last 100): -0.240, Epsilon: 0.869
Episode 14100, Avg Reward (last 100): -0.283, Epsilon: 0.868
Episode 14200, Avg Reward (last 100): -0.252, Epsilon: 0.868
Episode 14300, Avg Reward (last 100): -0.251, Epsilon: 0.867
Episode 14400, Avg Reward (last 100): -0.224, Epsilon: 0.866
Episode 14500, Avg Reward (last 100): -0.247, Epsilon: 0.865
Episode 14600, Avg Reward (last 100): -0.279, Epsilon: 0.864
Episode 14700, Avg Reward (last 100): -0.265, Epsilon: 0.863
Episode 14800, Avg Reward (last 100): -0.236, Epsilon: 0.862
Episode 14900, Avg Reward (last 100): -0.229, Epsilon: 0.862
Episode 15000, Avg Reward (last 100): -0.254, Epsilon: 0.861
Episode 15100, Avg Reward (last 100): -0.251, Epsilon: 0.860
Episode 15200, Avg Reward (last 100): -0.231, Epsilon: 0.859
Episode 15300, Avg Reward (last 100): -0.232, Epsilon: 0.858
Episode 15400, Avg Reward (last 100): -0.259, Epsilon: 0.857
Episode 15500, Avg Reward (last 100): -0.208, Epsilon: 0.856
Episode 15600, Avg Reward (last 100): -0.250, Epsilon: 0.856
Episode 15700, Avg Reward (last 100): -0.252, Epsilon: 0.855
Episode 15800, Avg Reward (last 100): -0.239, Epsilon: 0.854
Episode 15900, Avg Reward (last 100): -0.204, Epsilon: 0.853
Episode 16000, Avg Reward (last 100): -0.206, Epsilon: 0.852
Episode 16100, Avg Reward (last 100): -0.236, Epsilon: 0.851
Episode 16200, Avg Reward (last 100): -0.230, Epsilon: 0.850
Episode 16300, Avg Reward (last 100): -0.242, Epsilon: 0.850
Episode 16400, Avg Reward (last 100): -0.237, Epsilon: 0.849
Episode 16500, Avg Reward (last 100): -0.258, Epsilon: 0.848
Episode 16600, Avg Reward (last 100): -0.212, Epsilon: 0.847
Episode 16700, Avg Reward (last 100): -0.197, Epsilon: 0.846
Episode 16800, Avg Reward (last 100): -0.245, Epsilon: 0.845
Episode 16900, Avg Reward (last 100): -0.230, Epsilon: 0.844
Episode 17000, Avg Reward (last 100): -0.242, Epsilon: 0.844
Episode 17100, Avg Reward (last 100): -0.212, Epsilon: 0.843
Episode 17200, Avg Reward (last 100): -0.232, Epsilon: 0.842
Episode 17300, Avg Reward (last 100): -0.244, Epsilon: 0.841
Episode 17400, Avg Reward (last 100): -0.246, Epsilon: 0.840
Episode 17500, Avg Reward (last 100): -0.212, Epsilon: 0.839
Episode 17600, Avg Reward (last 100): -0.228, Epsilon: 0.839
Episode 17700, Avg Reward (last 100): -0.236, Epsilon: 0.838
Episode 17800, Avg Reward (last 100): -0.224, Epsilon: 0.837
Episode 17900, Avg Reward (last 100): -0.225, Epsilon: 0.836
Episode 18000, Avg Reward (last 100): -0.235, Epsilon: 0.835
Episode 18100, Avg Reward (last 100): -0.254, Epsilon: 0.834
Episode 18200, Avg Reward (last 100): -0.226, Epsilon: 0.834
Episode 18300, Avg Reward (last 100): -0.245, Epsilon: 0.833
Episode 18400, Avg Reward (last 100): -0.229, Epsilon: 0.832
Episode 18500, Avg Reward (last 100): -0.233, Epsilon: 0.831
Episode 18600, Avg Reward (last 100): -0.240, Epsilon: 0.830
Episode 18700, Avg Reward (last 100): -0.223, Epsilon: 0.829
Episode 18800, Avg Reward (last 100): -0.226, Epsilon: 0.829
Episode 18900, Avg Reward (last 100): -0.223, Epsilon: 0.828
Episode 19000, Avg Reward (last 100): -0.255, Epsilon: 0.827
Episode 19100, Avg Reward (last 100): -0.251, Epsilon: 0.826
Episode 19200, Avg Reward (last 100): -0.209, Epsilon: 0.825
Episode 19300, Avg Reward (last 100): -0.222, Epsilon: 0.824
Episode 19400, Avg Reward (last 100): -0.223, Epsilon: 0.824
Episode 19500, Avg Reward (last 100): -0.212, Epsilon: 0.823
Episode 19600, Avg Reward (last 100): -0.216, Epsilon: 0.822
Episode 19700, Avg Reward (last 100): -0.253, Epsilon: 0.821
Episode 19800, Avg Reward (last 100): -0.218, Epsilon: 0.820
Episode 19900, Avg Reward (last 100): -0.225, Epsilon: 0.820
"""