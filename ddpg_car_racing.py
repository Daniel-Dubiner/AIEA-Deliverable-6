import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# --- Hyperparameters ---
ENV_NAME = "CarRacing-v3"
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
EPISODES = 50
MAX_STEPS = 1000
NOISE_STD = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# --- DDPG Agent ---
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise:
            action += np.random.normal(0, NOISE_STD, size=action.shape)
        return np.clip(action, -1, 1)

    def train(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + GAMMA * (1 - dones) * self.critic_target(next_states, next_actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# --- Preprocessing ---
def preprocess(obs):
    # Convert image to grayscale and flatten
    gray = np.mean(obs, axis=2) / 255.0
    return gray.flatten()

# --- Main Training Loop ---
def main():
    env = gym.make(ENV_NAME)
    obs, _ = env.reset()
    state_dim = preprocess(obs).shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    agent = DDPGAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(next_obs)
            done = terminated or truncated

            agent.buffer.add(state, action, reward, next_state, float(done))
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{EPISODES} | Reward: {total_reward:.2f}")

    env.close()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.plot(np.convolve(episode_rewards, np.ones(5)/5, mode='valid'), label="5-ep Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDPG on CarRacing-v3")
    plt.legend()
    plt.savefig("/home/ubuntu/persistent/ddpg_results.png")
    plt.close()
    print("Plot saved to ddpg_results.png")

if __name__ == "__main__":
    main()
