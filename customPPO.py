import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layer(x)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

class PPOTrainer:
    def __init__(self, env_fn, input_dim, action_dim, gamma=0.99, lr=3e-4, clip_epsilon=0.2, update_epochs=4, batch_size=64):
        self.device = torch.device("cpu")
        self.env_fn = env_fn
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.model = ActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_returns(self, rewards, values, dones):
        returns = []
        R = 0
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def train(self, max_episodes):
        env = self.env_fn()
        total_rewards = []

        for episode in range(max_episodes):
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            log_probs, values, rewards, states, actions, dones = [], [], [], [], [], []

            done = False
            while not done:
                policy_logits, value = self.model(obs)
                probs = Categorical(logits=policy_logits)
                action = probs.sample()

                states.append(obs)
                actions.append(action)
                log_probs.append(probs.log_prob(action))
                values.append(value.squeeze())

                obs, reward, terminated, truncated, _ = env.step(action.item())
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                rewards.append(reward)
                dones.append(terminated or truncated)
                done = terminated or truncated

            states = torch.cat(states)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)

            returns = self.compute_returns(rewards, values, dones)
            advantages = returns - values

            for _ in range(self.update_epochs):
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_states = states[start:end]
                    batch_actions = actions[start:end]
                    batch_log_probs = log_probs[start:end]
                    batch_advantages = advantages[start:end]
                    batch_returns = returns[start:end]

                    policy_logits, value = self.model(batch_states)
                    probs = Categorical(logits=policy_logits)

                    new_log_probs = probs.log_prob(batch_actions)
                    entropy = probs.entropy().mean()

                    ratio = (new_log_probs - batch_log_probs).exp()
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                    actor_loss = -torch.min(surrogate1, surrogate2).mean()
                    critic_loss = (batch_returns - value.squeeze()).pow(2).mean()
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_rewards.append(sum(rewards))
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")

        return total_rewards

    def evaluate_policy(self, n_episodes):
        env = self.env_fn()
        self.model.eval()
        total_rewards = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            done = False
            episode_reward = 0

            while not done:
                with torch.no_grad():
                    policy_logits, _ = self.model(obs)
                    probs = Categorical(logits=policy_logits)
                    action = probs.sample()

                obs, reward, terminated, truncated, _ = env.step(action.item())
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode}, Total Reward: {episode_reward}")

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {n_episodes} Episodes: {avg_reward}")
        return avg_reward

# Example Usage
def make_env():
    return gym.make("CartPole-v1")

trainer = PPOTrainer(env_fn=make_env, input_dim=4, action_dim=2)
trainer.train(max_episodes=1000)
trainer.evaluate_policy(n_episodes=10)
