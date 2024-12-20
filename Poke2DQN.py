import asyncio
import numpy as np
import torch
from stable_baselines3 import DQN
from typing import Callable
from gymnasium.spaces import Box
from poke_env.player import Gen9EnvSinglePlayer, Player
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update
from poke_env.data import GenData
from FusionBot import FusionPlayer
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_rewards(log_file_path):
    """
    Plots the mean reward per episode from the Monitor log.

    :param log_file_path: Path to the log file generated by the Monitor.
    """
    # Load the log file
    rewards = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            rewards.append(float(parts[1]))  # The second column is the reward

    # Calculate mean reward per episode
    mean_rewards = [sum(rewards[:i+1]) / (i+1) for i in range(len(rewards))]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards, label="Mean Reward Per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward Per Episode During Training")
    plt.legend()
    plt.grid()
    plt.show()


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


class DQNBot(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_data = GenData.from_gen(9)

    def embed_battle(self, battle):
        if isinstance(battle, tuple):
            battle = battle[0]
        if isinstance(battle, np.ndarray):
            return np.zeros(10, dtype=np.float32)

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

    def describe_embedding(self):
        low = np.array([-1.0] * 4 + [0.0] * 4 + [0.0, 0.0], dtype=np.float32)
        high = np.array([1.0] * 4 + [4.0] * 4 + [1.0, 1.0], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def calc_reward(self, last_state, current_state):
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=5
        )

    def reset(self, seed=None, options=None):
        initial_obs = super().reset(seed=seed, options=options)
        observation = self.embed_battle(initial_obs)
        return observation, {}

    def step(self, action):
        step_output = super().step(action)
        if len(step_output) == 4:
            battle, reward, done, info = step_output
            terminated, truncated = done, False
        elif len(step_output) == 5:
            battle, reward, terminated, truncated, info = step_output
        else:
            raise ValueError(f"Unexpected step output: {step_output}")

        observation = self.embed_battle(battle)
        return observation, reward, terminated, truncated, info

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.utils import polyak_update

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Train the model for a given number of gradient steps using the DDQN update rule.
        """
        # Train for the given number of gradient steps
        for _ in range(gradient_steps):
            # Sample a batch from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                # Get the actions from the online network (main network)
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                # Evaluate the value of those actions using the target network
                next_q_values = self.q_net_target(replay_data.next_observations).gather(1, next_actions)
                # Compute the target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get the current Q-values
            current_q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions)
            
            # Compute the loss
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # Optimize the Q-network
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

        # Update the target network
        if self.target_update_interval > 1:
            if self.num_timesteps % self.target_update_interval == 0:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), 1.0)
        else:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)



def linear_schedule(initial_value: float, final_value: float, max_steps: int) -> Callable[[float], float]:
    """
    Returns a function that computes a value linearly decreasing over time.

    :param initial_value: The initial value at step 0.
    :param final_value: The final value at step max_steps.
    :param max_steps: The total number of steps.
    """
    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule

async def main():
    np.random.seed(42)
    torch.manual_seed(42)

    opponent = FusionPlayer()
    envOG_player = DQNBot(opponent=opponent)
    env_player = Monitor(envOG_player, filename="DQN_training_log.csv")
    env_player = DummyVecEnv([lambda: env_player])

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     model = DQN(
#     "MlpPolicy",
#     env_player,
#     verbose=1,
#     gamma=0.99,  # Increased discount factor
#     learning_rate=linear_schedule(0.001, 0.0001, 200000),  # Slightly higher initial learning rate
#     buffer_size=100000,
#     batch_size=256,  # Increased batch size
#     learning_starts=10000,  # Extended warmup period
#     exploration_initial_eps=1.0,  # Full initial exploration
#     exploration_final_eps=0.01,
#     exploration_fraction=0.2,  # Extended exploration period
#     target_update_interval=500,  # Less frequent but more stable target updates
#     train_freq=1,
#     gradient_steps=1,
#     seed=42,
#     device=device
# )
    model = DoubleDQN(
        "MlpPolicy",
        env_player,
        verbose=1,
        gamma=0.99,
        learning_rate=linear_schedule(0.001, 0.0001, 200000),
        buffer_size=100000,
        batch_size=256,
        learning_starts=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.2,
        target_update_interval=500,
        train_freq=1,
        gradient_steps=1,
        seed=42,
        # device=device
    )

    model.learn(total_timesteps=100000)  # Train the agent
    print("#### Training against Max Damage Bot #####")
    print("Won", envOG_player.n_won_battles, "total battles", envOG_player.n_finished_battles)

    #plot_rewards("DQN_training_log.csv.monitor.csv")



if __name__ == "__main__":
    asyncio.run(main())
