import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
from typing import List, Tuple

from gymnasium.spaces import Box
from poke_env.player import Gen9EnvSinglePlayer
from poke_env.data import GenData
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rp import RandomPlayer
from FusionBot import FusionPlayer

import matplotlib.pyplot as plt

class A3CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A3CNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(64, output_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        policy = self.actor(shared_features)
        value = self.critic(shared_features)
        return policy, value

class A3CAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.network = A3CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.network(state_tensor)
        policy = torch.softmax(policy, dim=-1)
        action_dist = Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    
    def compute_returns(self, rewards, dones, values):
        returns = []
        R = 0
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def compute_loss(self, log_probs, returns, values):
        # Advantage calculation
        advantages = torch.tensor(returns) - torch.tensor(values)
        
        # Actor loss (policy gradient)
        actor_loss = -(torch.tensor(log_probs) * advantages.detach()).mean()
        
        # Critic loss (value approximation)
        critic_loss = advantages.pow(2).mean()
        
        # Entropy bonus to encourage exploration
        entropy_loss = -(torch.tensor(log_probs) * torch.exp(torch.tensor(log_probs))).mean()
        
        return actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

class A3CPokemonBot(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_data = GenData.from_gen(9)

    def embed_battle(self, battle):
        # Similar to your PPO implementation's embedding
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)

        moves = battle.available_moves if hasattr(battle, 'available_moves') else []
        for i, move in enumerate(moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=self.gen_data.type_chart
                )

        remaining_mon_team = len([mon for mon in battle.team.values() if not mon.fainted]) / 6
        remaining_mon_opponent = len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6

        return np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            [remaining_mon_team, remaining_mon_opponent]
        ])
    def calc_reward(self, last_state, current_state):
        """
        Calculate reward based on battle progression.
        Mimics the reward computing helper from PPO implementation.
        """
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=5
        )

    def describe_embedding(self):
        low = np.array([-1.0] * 4 + [0.0] * 4 + [0.0, 0.0], dtype=np.float32)
        high = np.array([1.0] * 4 + [4.0] * 4 + [1.0, 1.0], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

def worker(shared_agent, worker_agent, opponent, total_episodes=1000):
    env = A3CPokemonBot(opponent=opponent)
    state_dim = len(env.embed_battle(env.reset()[0]))
    action_dim = len(env.action_space)

    for episode in range(total_episodes):
        state, _ = env.reset()
        done = False
        
        episode_rewards = []
        log_probs = []
        values = []
        
        while not done:
            # Select action
            action, log_prob = worker_agent.select_action(state)
            
            # Get policy and state value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, state_value = shared_agent.network(state_tensor)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            episode_rewards.append(reward)
            log_probs.append(log_prob)
            values.append(state_value.item())
            
            state = next_state
        
        # Compute returns and update shared network
        returns = worker_agent.compute_returns(episode_rewards, [done], values)
        
        # Compute loss and update network
        loss = worker_agent.compute_loss(log_probs, returns, values)
        
        shared_agent.optimizer.zero_grad()
        loss.backward()
        shared_agent.optimizer.step()

def main():
    # Ensure multiprocessing method is set correctly
    mp.set_start_method('fork')  # Use 'fork' for macOS
    
    # Initialize shared and worker networks
    shared_agent = A3CAgent(
        state_dim=10,  # Matches embedding from embed_battle
        action_dim=4,  # Assuming 4 possible actions
        learning_rate=1e-3,
        gamma=0.99
    )
    
    # Multiprocessing to create multiple worker agents
    num_workers = 4
    processes = []
    
    for _ in range(num_workers):
        worker_agent = A3CAgent(
            state_dim=10,
            action_dim=4,
            learning_rate=1e-3,
            gamma=0.99
        )
        
        # Share network parameters
        worker_agent.network.load_state_dict(shared_agent.network.state_dict())
        
        opponent = RandomPlayer(battle_format="gen9randombattle")
        
        p = mp.Process(target=worker, args=(shared_agent, worker_agent, opponent))
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Save the final model
    torch.save(shared_agent.network.state_dict(), 'model/A3C/A3C_Pokemon_model.pt')
    
    # Optional: Evaluation
    test_opponent = FusionPlayer(battle_format="gen9randombattle")
    test_env = A3CPokemonBot(opponent=test_opponent)
    
    # Perform evaluation logic here
    print("Evaluation results would be added here")

if __name__ == "__main__":
    main()