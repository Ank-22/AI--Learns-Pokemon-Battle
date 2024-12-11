import asyncio
import numpy as np
import torch
from stable_baselines3 import PPO

from gymnasium.spaces import Box
from poke_env.player import Gen9EnvSinglePlayer, Gen8EnvSinglePlayer,Gen4EnvSinglePlayer, Player
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
from FusionBot import FusionPlayer
from rp import MaxDamagePlayer, RandomPlayer
from poke_env.data import GenData
from customA3c import A3CTrainer
import torch



class RewardLoggerCallback(BaseCallback):
    """
    Custom callback for logging mean episodic rewards and timesteps during training.
    """
    def __init__(self, mean_rewards_list, timesteps_list, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.mean_rewards_list = mean_rewards_list
        self.timesteps_list = timesteps_list

    def _on_step(self) -> bool:
        # Log mean episodic rewards if available in infos
        if self.locals["infos"]:
            for info in self.locals["infos"]:
                if "episode" in info and "r" in info["episode"]:
                    self.mean_rewards_list.append(info["episode"]["r"])
                    self.timesteps_list.append(self.num_timesteps)  # Log the current timestep
        return True

def smooth_data(data, window_size=50):
    """
    Smooths the data using a simple moving average.
    :param data: List of values to smooth.
    :param window_size: Number of points to average over.
    :return: Smoothed data as a numpy array.
    """
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    return smoothed



# Plotting mean rewards
def plot_rewards(mean_rewards, timesteps, window_size=50):
    """
    Plots smoothed mean reward over training timesteps.
    :param mean_rewards: List of mean rewards.
    :param timesteps: List of corresponding timesteps.
    :param window_size: Window size for smoothing.
    """
    smoothed_rewards = smooth_data(mean_rewards, window_size=window_size)
    smoothed_timesteps = timesteps[:len(smoothed_rewards)]  # Align timesteps with smoothed rewards

    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_timesteps, smoothed_rewards, label=f"Smoothed Mean Reward (window={window_size})")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"Mean Episodic Reward of PPO over {timesteps[-1]} Timesteps")
    plt.legend()
    plt.grid()
    plt.show()



# Environment definition
class A3CBot(Gen4EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_data = GenData.from_gen(4)


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


# PPO training function
async def main():
    np.random.seed(42)
    torch.manual_seed(42)

    opponent = RandomPlayer( battle_format="gen4randombattle")  # Substitute this with your actual opponent bot
    envOG_player = A3CBot(opponent=opponent)
    
    def env_fn():
        return A3CBot(opponent=RandomPlayer(battle_format="gen4randombattle"))  # Create a new environment instance for each worker

    a3c_trainer = A3CTrainer(
        env_fn=env_fn,
        input_dim=envOG_player.describe_embedding().shape[0],
        action_dim=envOG_player.action_space.n,
        num_workers=4,  # Adjust based on your system
        gamma=0.99,
        lr=1e-3,
        reward_file="a3c_rewards_gen4.csv"  # Save rewards to this file
    )


    rewards = a3c_trainer.train(max_episodes=1000)

    # Plot the rewards
    print("#### Training against Opponent Bot #####")
    print("Won", envOG_player.n_won_battles, "total battles", envOG_player.n_finished_battles)

    model_save_path = "model/A3C/a3c_model_gen4.pth"
    torch.save(a3c_trainer.global_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    a3c_trainer.evaluate_policy(n_episodes=100)

    # second_opponent = FusionPlayer(battle_format="gen9randombattle")
    # testOG_player = PPOBot(opponent=second_opponent)
    # testOG_player.reset_env(restart=True, opponent=second_opponent)
    # test_player = Monitor(testOG_player, filename=None)
    # test_player = DummyVecEnv([lambda: test_player])
    # print("Results against max base power player:")
    # mean_reward, std_reward = evaluate_policy(model, test_player, n_eval_episodes=100, return_episode_rewards=False)
    # print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    # print(
    #     f"PPO Evaluation: {testOG_player.n_won_battles} victories out of {testOG_player.n_finished_battles} episodes"
    # )


if __name__ == "__main__":
    asyncio.run(main())