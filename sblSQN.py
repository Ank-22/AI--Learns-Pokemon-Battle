import asyncio
import numpy as np
import torch
from stable_baselines.deepq import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.monitor import Monitor
from stable_baselines.common.policies import MlpPolicy
from typing import Callable
from gymnasium.spaces import Box
from poke_env.player import Player, Gen9EnvSinglePlayer
from poke_env.data import GenData
from FusionBot import FusionPlayer
import matplotlib.pyplot as plt


def plot_rewards(log_file_path):
    """
    Plots the mean reward per episode from the Monitor log.

    :param log_file_path: Path to the log file generated by the Monitor.
    """
    rewards = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            rewards.append(float(parts[1]))  # The second column is the reward

    mean_rewards = [sum(rewards[:i + 1]) / (i + 1) for i in range(len(rewards))]

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


def linear_schedule(initial_value: float, final_value: float, max_steps: int) -> Callable[[float], float]:
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

    model = DQN(
        policy=MlpPolicy,
        env=env_player,
        verbose=1,
        gamma=0.99,
        learning_rate=linear_schedule(0.001, 0.0001, 200000),
        buffer_size=100000,
        batch_size=256,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        train_freq=1,
        target_network_update_freq=500,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
    )

    model.learn(total_timesteps=100000)
    print("#### Training against Max Damage Bot #####")
    print("Won", envOG_player.n_won_battles, "total battles", envOG_player.n_finished_battles)


if __name__ == "__main__":
    asyncio.run(main())
