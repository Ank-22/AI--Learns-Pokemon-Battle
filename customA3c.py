import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from threading import Thread
from queue import Queue
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

class A3CTrainer:
    def __init__(self, env_fn, input_dim, action_dim, num_workers=4, gamma=0.99, lr=1e-4, reward_file="rewards.csv"):
        self.device = torch.device("cpu") #torch.device("mps") if torch.backends.mps.is_available() else 
        self.env_fn = env_fn
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_workers = num_workers
        self.gamma = gamma
        self.global_model = ActorCritic(input_dim, action_dim).to(self.device)
        self.global_model.share_memory()
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=lr)
        self.reward_file = reward_file

    def worker_function(self, worker_id, max_episodes, reward_queue):
        env = self.env_fn()
        local_model = ActorCritic(self.input_dim, self.action_dim).to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())

        for episode in range(max_episodes):
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            done = False
            log_probs = []
            rewards = []
            values = []

            while not done:
                policy_logits, value = local_model(obs)
                probs = Categorical(logits=policy_logits)
                action = probs.sample()
                log_probs.append(probs.log_prob(action))
                values.append(value)

                obs, reward, terminated, truncated, _ = env.step(action.item())
                obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                rewards.append(reward)
                done = terminated or truncated

            R = 0 if done else local_model(obs)[1].item()
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(self.device)
            log_probs = torch.stack(log_probs)
            values = torch.stack(values).squeeze()

            # Advantage computation
            advantage = returns - values
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + critic_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            for global_param, local_param in zip(self.global_model.parameters(), local_model.parameters()):
                global_param.grad = local_param.grad
            self.optimizer.step()
            local_model.load_state_dict(self.global_model.state_dict())

            reward_queue.put(sum(rewards))
            print(f"Worker {worker_id}, Episode {episode}, Total Reward: {sum(rewards)}")

    def train(self, max_episodes):
        reward_queue = Queue()
        workers = []
        for worker_id in range(self.num_workers):
            worker = Thread(target=self.worker_function, args=(worker_id, max_episodes // self.num_workers, reward_queue))
            worker.start()
            workers.append(worker)

        total_rewards = []
        while any(worker.is_alive() for worker in workers):
            while not reward_queue.empty():
                total_rewards.append(reward_queue.get())

        for worker in workers:
            worker.join()

        with open(self.reward_file, "w") as f:
            for reward in total_rewards:
                f.write(f"{reward}\n")

        return total_rewards

    def evaluate_policy(self, n_episodes):
        """Evaluate the policy using a single worker over n episodes."""
        env = self.env_fn()
        local_model = ActorCritic(self.input_dim, self.action_dim).to(self.device)
        local_model.load_state_dict(self.global_model.state_dict())
        local_model.eval()

        total_rewards = []
        for episode in range(n_episodes):
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            done = False
            episode_reward = 0

            while not done:
                with torch.no_grad():
                    policy_logits, _ = local_model(obs)
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
        print("Won", env.n_won_battles, "total battles", env.n_finished_battles)
        return avg_reward
