#!/usr/bin/env python
# coding: utf-8

# # FrozenLake [8x8 | random map | no slip]

# In[14]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import cv2
from PIL import Image

plt.style.use(['dark_background', 'seaborn-v0_8'])

# seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch device:", device)


# ## export raw training img -> video

# In[15]:


import os


def count_files(dir_path: str) -> int:
    return sum(
        1 for entry in os.scandir(dir_path)
        if entry.is_file()
    )


f"files/pics/{count_files("files/pics/") + 1:04d}.png"


# In[32]:


# !rm files/pics/*


# In[ ]:


# !ffmpeg -framerate 2 -i files/pics/%04d.png -c:v libx264 -pix_fmt yuv420p files/output-1.mp4


# ## Img Wrapper

# In[33]:


class FrozenLakeImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, img_size=(84, 84), grayscale=False):
        super().__init__(env)
        self.img_size = img_size
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(*img_size, channels),
            dtype=np.uint8
        )

    def observation(self, _):
        frame = self.env.render()
        if frame is None:
            channels = 1 if self.grayscale else 3
            return np.zeros((*self.img_size, channels), dtype=np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(frame)

        # Grayscale conversion
        if self.grayscale:
            img = img.convert('L')

        # Resize
        img = img.resize(self.img_size, Image.NEAREST)
        img.save(f"files/pics/{count_files("files/pics/") + 1:04d}.png")  #!DEBUG!#
        obs = np.array(img, dtype=np.uint8)

        # Add channel dimension for grayscale
        if self.grayscale:
            obs = np.expand_dims(obs, axis=-1)

        return obs


def preprocess_frame(frame, resize=(84, 84), grayscale=True):
    """For testing preprocessing"""
    img = Image.fromarray(frame)
    if grayscale:
        img = img.convert('L')
    img = img.resize(resize, Image.NEAREST)
    return np.array(img)


# In[34]:


class RandomMapResetWrapper(gym.Wrapper):
    def __init__(self, env, size=8, p=0.8, change_every=1):
        super().__init__(env)
        self.size = size
        self.p = p
        self.change_every = change_every
        self.episode_count = 0
        self.current_desc = None
        self._make_env_with_random_map()

    def _make_env_with_random_map(self):
        new_desc = generate_random_map(size=self.size, p=self.p)
        self.current_desc = tuple(new_desc)
        self.env = gym.make(
            "FrozenLake-v1",
            desc=new_desc,
            is_slippery=self.env.spec.kwargs.get("is_slippery", False),
            render_mode=self.env.render_mode
        )

    def reset(self, **kwargs):
        self.episode_count += 1
        if (self.episode_count - 1) % self.change_every == 0:
            # Time to change map
            self._make_env_with_random_map()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


# In[35]:


class FrozenLakeCNN(nn.Module):
    def __init__(self, action_size, grayscale=True):
        super().__init__()
        self.grayscale = grayscale
        channels = 1 if grayscale else 3

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate feature size: ((84-8)//4 + 1) = 20 → ((20-4)//2 +1)=9 → (9-3+1)=7
        self.fc = nn.Sequential(
            nn.Linear(64 * 18 * 18, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        # x shape: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.conv(x))


# In[36]:


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


# In[37]:


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


# In[38]:


def create_env(render_mode=None, use_image=False, img_size=(84, 84), grayscale=True,
               random_map_every_reset=True, change_every=1):
    env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode=render_mode)
    if random_map_every_reset:
        env = RandomMapResetWrapper(env, size=8, p=0.8, change_every=change_every)
    if use_image:
        env = FrozenLakeImageWrapper(env, img_size=img_size, grayscale=grayscale)
        state_shape = (*img_size, 1 if grayscale else 3)
    else:
        state_shape = env.observation_space.n
    action_size = env.action_space.n
    return env, state_shape, action_size


def train_dqn(agent, env, state_shape, action_size,
              episodes=2000,
              max_steps=100,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=0.995, ):
    global history_rewards, history_epsilons

    scores = deque(maxlen=100)
    recent_maps = deque(maxlen=100)

    epsilon = epsilon_start
    print("Starting training...")

    for episode in range(episodes):
        state, _ = env.reset()
        # state is now image - no one-hot needed
        total_reward = 0

        current_map = getattr(env.env, 'current_desc', None)
        recent_maps.append(current_map)

        for t in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # --- REWARD SHAPING ---
            shaped_reward = reward
            if not truncated and terminated:
                if reward == 1.0:
                    shaped_reward = 1.0  # goal
                else:
                    shaped_reward = -1.0  # hole
            else:
                # # Penalize stuck moves (only in discrete mode!)
                # print('-~>', next_state, state)
                # if next_state == state:
                #     shaped_reward += -0.1  # strong penalty for no-op
                # else:
                shaped_reward += -0.01  # small step penalty

            agent.remember(state, action, shaped_reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += shaped_reward

            if done:
                break

        scores.append(total_reward)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        history_rewards.append(np.mean(scores))
        history_epsilons.append(epsilon)

        if episode % 100 == 0:
            unique_maps = len(set(m for m in recent_maps if m is not None))
            print(f"Episode {episode}, Avg Reward: {np.mean(scores):.3f}, "
                  f"Epsilon: {epsilon:.3f}, Unique Maps (last 100): {unique_maps}")

        if episode % 1500 == 0 and episode > 0:
            render_env, _, _ = create_env(render_mode="human", use_image=True, random_map_every_reset=True)
            evaluate_agent(agent, render_env, episodes=2, no_log=True)
            render_env.close()

    return agent, env


def evaluate_agent(agent, env, episodes=10, max_steps=1000, no_log=False):
    """
    Evaluate agent on pixel-based environment.
    - Works with both image and discrete state spaces
    """
    success = 0
    for _ in range(episodes):
        state, _ = env.reset()
        for step in range(max_steps):
            action = agent.act(state, epsilon=0.0)  # Greedy policy
            state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                if reward == 1.0:  # Reached goal
                    success += 1
                break

    if not no_log:
        print(f"\nSuccess rate: {success}/{episodes} ({100 * success / episodes:.1f}%)")
    return success


def act(self, state, epsilon=0.0):
    if random.random() < epsilon:
        return random.randrange(self.action_size)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = self.q_net(state_tensor)
    return q_values.argmax().item()


# ## training loop

# In[41]:


_, _, action_size = create_env(use_image=False)
env, state_shape, _ = create_env(
    render_mode="rgb_array",
    use_image=True,
    img_size=(84, 84),
    grayscale=True,
    random_map_every_reset=True,
    change_every=2_000)

agent = DQNAgent(
    modelClass=lambda s, a: FrozenLakeCNN(a, grayscale=grayscale),
    state_size=None,  # Not used
    action_size=action_size,
    lr=1e-4,
    gamma=0.99,
    buffer_size=20_000,
    batch_size=2_000,
    target_update=100
)

history_rewards, history_epsilons = [], []


# In[42]:


try:
    agent, env = train_dqn(
        agent=agent,
        env=env,
        state_shape=state_shape,
        action_size=action_size,
        episodes=2_000,
        max_steps=500,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,
    )
finally:
    # Plotting code (unchanged)
    print(f"Model: {FrozenLakeCNN.__name__}")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Avg Reward (last 100)', color=color)
    ax1.plot(history_rewards, color=color, label='Avg Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    if len(history_rewards) > 0:
        y_min = np.floor(min(history_rewards) * 10) / 10
        y_max = np.ceil(max(history_rewards) * 10) / 10
        ax1.set_yticks(np.arange(y_min, y_max + 0.05, 0.1))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(history_epsilons, color=color, label='Epsilon', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Progress')
    fig.tight_layout()
    plt.show()


# In[ ]:


eval_env, _, _ = create_env(render_mode="human", use_image=True, random_map_every_reset=True)
evaluate_agent(agent, eval_env, episodes=5)
eval_env.close()


# It isnt working at all.
# Trying MLP

# # Approach 2: using MLP

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import os
from PIL import Image
import shutil

plt.style.use(['dark_background', 'seaborn-v0_8'])

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)


# In[2]:


def count_files(dir_path: str) -> int:
    os.makedirs(dir_path, exist_ok=True)
    return sum(1 for entry in os.scandir(dir_path) if entry.is_file())

def clear_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# In[3]:


from collections import deque as dq

def is_solvable(desc):
    size = len(desc)
    grid = [list(row) for row in desc]
    if grid[size-1][size-1] == 'H':
        return False
    visited = [[False]*size for _ in range(size)]
    q = dq([(0, 0)])
    visited[0][0] = True
    while q:
        x, y = q.popleft()
        if (x, y) == (size-1, size-1):
            return True
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny] and grid[nx][ny] != 'H':
                visited[nx][ny] = True
                q.append((nx, ny))
    return False

def generate_solvable_map(size=8, p=0.85, max_attempts=50):
    for _ in range(max_attempts):
        desc = generate_random_map(size=size, p=p)
        if is_solvable(desc):
            return desc
    # Fallback
    return generate_random_map(size=size, p=p)


# In[4]:


class FrozenLakeFullMapWrapper(gym.ObservationWrapper):
    def __init__(self, env, img_size=(84, 84), record_dir="files/pics"):
        super().__init__(env)
        self.img_size = img_size
        self.record_dir = record_dir
        self.episode_frame_count = 0
        # Output: (H, W, 3) → [grayscale, x_coord, y_coord]
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=255.0,
            shape=(*img_size, 3),
            dtype=np.float32
        )
        os.makedirs(record_dir, exist_ok=True)

    def observation(self, _):
        frame = self.env.render()
        if frame is None:
            return np.zeros((*self.img_size, 3), dtype=np.float32)

        # Resize to target
        img = Image.fromarray(frame).resize(self.img_size, Image.NEAREST)
        gray = np.array(img.convert('L'), dtype=np.float32)  # (H, W)

        H, W = self.img_size
        x = np.linspace(-1, 1, W, dtype=np.float32)
        y = np.linspace(-1, 1, H, dtype=np.float32)
        x_ch = np.tile(x, (H, 1))        # (H, W)
        y_ch = np.tile(y[:, None], (1, W))  # (H, W)

        # Stack: [gray, x, y] → (H, W, 3)
        obs = np.stack([gray, x_ch, y_ch], axis=-1)

        # Save frame for video (only during training episodes)
        if hasattr(self, '_recording') and self._recording:
            img.save(os.path.join(self.record_dir, f"{self.episode_frame_count:05d}.png"))
            self.episode_frame_count += 1

        return obs

    def start_recording(self):
        self._recording = True
        self.episode_frame_count = 0

    def stop_recording(self):
        self._recording = False


# In[5]:


class RandomMapResetWrapper(gym.Wrapper):
    def __init__(self, env, size=8, p=0.85, change_every=1):
        super().__init__(env)
        self.size = size
        self.p = p
        self.change_every = change_every
        self.episode_count = 0
        self.current_desc = None
        self._make_env_with_random_map()

    def _make_env_with_random_map(self):
        new_desc = generate_solvable_map(size=self.size, p=self.p)
        self.current_desc = tuple(new_desc)
        self.env = gym.make(
            "FrozenLake-v1",
            desc=new_desc,
            is_slippery=False,
            render_mode="rgb_array"  # ← must render!
        )

    def reset(self, **kwargs):
        self.episode_count += 1
        if (self.episode_count - 1) % self.change_every == 0:
            old_desc = self.current_desc
            self._make_env_with_random_map()
            print(f"🆕 New map (ep {self.episode_count}): changed from {old_desc[:2]}... to {self.current_desc[:2]}...")
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


# In[6]:


class FrozenLakeFullMapCNN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # Input: (B, 3, H, W) → [gray, x, y]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # For 84x84 → 18x18 → 64*18*18 = 20736
        self.fc = nn.Sequential(
            nn.Linear(64 * 18 * 18, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        # x: (B, H, W, 3) → permute to (B, 3, H, W)
        x = x.permute(0, 3, 1, 2)
        x[:, 0] = x[:, 0] / 255.0  # normalize grayscale only
        # x[:, 1:] already in [-1,1]
        return self.fc(self.conv(x))


# In[7]:


# (Same as before — uses FloatTensor for image inputs)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, modelClass, action_size, lr=1e-4, gamma=0.99,
                 buffer_size=20000, batch_size=64, target_update=100):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.q_net = modelClass(action_size).to(device)
        self.target_net = modelClass(action_size).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

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
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# In[8]:


def create_env(render_mode="rgb_array", record_dir="files/pics", change_every=500):
    env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode=render_mode)
    env = RandomMapResetWrapper(env, size=8, p=0.85, change_every=change_every)
    env = FrozenLakeFullMapWrapper(env, img_size=(84, 84), record_dir=record_dir)
    state_shape = (84, 84, 3)
    action_size = env.action_space.n
    return env, state_shape, action_size

def train_dqn(agent, env, state_shape, action_size,
              episodes=5000,
              max_steps=200,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=0.995,
              record_every=1000,
              change_every=100):
    global history_rewards, history_epsilons
    scores = deque(maxlen=100)
    epsilon = epsilon_start
    print("🚀 Starting training with full map observation...")

    for episode in range(episodes):
        # Record every `record_every` episodes
        if episode % record_every == 0:
            env.start_recording()
        else:
            env.stop_recording()

        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward shaping
            if not truncated and terminated:
                shaped_reward = 1.0 if reward == 1.0 else -1.0
            else:
                shaped_reward = -0.01

            agent.remember(state, action, shaped_reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += shaped_reward
            if done:
                break

        # Stop recording after episode
        env.stop_recording()

        scores.append(total_reward)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        history_rewards.append(np.mean(scores))
        history_epsilons.append(epsilon)

        if episode % 100 == 0:
            print(f"Ep {episode:4d} | Avg R: {np.mean(scores):6.3f} | ε: {epsilon:.3f}")

        # Compile video after recording episode
        if episode % record_every == 0 and episode > 0:
            video_path = f"files/output_ep{episode:05d}.mp4"
            os.system(f"ffmpeg -y -framerate 5 -i files/pics/%05d.png "
                      f"-c:v libx264 -pix_fmt yuv420p {video_path} > /dev/null 2>&1")
            print(f"🎥 Video saved: {video_path}")
            clear_dir("files/pics")  # reset for next recording

        # Inside train_dqn loop
        if (episode > 0) and (episode % change_every == 0):
            epsilon = max(epsilon, 0.5)

    return agent, env


# ## Training loop

# In[9]:


# Clear old frames
clear_dir("files/pics")

# Create env (with full map observation)
env, state_shape, action_size = create_env(change_every=2000)

agent = DQNAgent(
    modelClass=FrozenLakeFullMapCNN,
    action_size=action_size,
    lr=1e-4,
    gamma=0.99,
    buffer_size=20000,
    batch_size=64,
    target_update=100
)

history_rewards, history_epsilons = [], []


# In[ ]:


try:
    agent, env = train_dqn(
        agent=agent,
        env=env,
        state_shape=state_shape,
        action_size=action_size,
        episodes=3000,
        max_steps=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        record_every=50,
        change_every=2000,
    )
finally:
    # Final plot
    plt.figure(figsize=(10,5))
    plt.plot(history_rewards, label='Avg Reward (100 eps)')
    plt.xlabel('Episode'); plt.ylabel('Reward')
    plt.title('Training with Full Map Observation')
    plt.grid(True, alpha=0.3)
    plt.show()


# In[ ]:





# In[ ]:


# Evaluate on new random maps
print("\n🔍 Final Evaluation on 5 new random maps:")
eval_env, _, _ = create_env(render_mode=None, random_map_every_reset=True, change_every=1)
evaluate_agent(agent, eval_env, episodes=5)
eval_env.close()

