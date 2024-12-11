import os

import asyncio
import numpy as np
import torch
from gymnasium.spaces import Box
from poke_env.player import Gen9EnvSinglePlayer, Player

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import DummyVecEnv
from FusionBot import FusionPlayer
from poke_env.data import GenData
from PokePPO import PPOBot
from rp import RandomPlayer, MaxDamagePlayer

# Define the path to your saved model
model_path = 'model/PPO/lr3_ns_2048_100K_Gen4.pt'

# Define your environment setup (replace with your custom environment)
def make_env():
    # Assuming testOG_player is your custom environment setup
    #second_opponent = MaxDamagePlayer( battle_format="gen4randombattle")
    testOG_player = PPOBot()
    testOG_player.send_challenges("zxcvbnm22", n_challenges=10)

    return testOG_player

# Create and vectorize the environment
player = make_env()
env = DummyVecEnv([lambda: player])

# Load the model
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Evaluate the model
print("Evaluating the PPO model...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

print(
        f"PPO Evaluation: {player.n_won_battles} victories out of {player.n_finished_battles} episodes"
    )
