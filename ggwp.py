import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class SimpleRLPlayer(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_battle(self, battle):
        """
        Convert the current battle state into a numerical representation suitable for RL.
        """
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)

        # Process available moves
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100  # Rescale for learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart
                )

        # Count non-fainted Pokémon
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if not mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
        )

        # Construct the observation vector
        embedding = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )
        return embedding

    def describe_embedding(self):
        """
        Define the observation space based on the shape and range of embed_battle's output.
        """
        low = np.array([-1.0] * 4 + [0.0] * 4 + [0.0, 0.0], dtype=np.float32)
        high = np.array([1.0] * 4 + [4.0] * 4 + [1.0, 1.0], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def calc_reward(self, last_state, current_state) -> float:
        """
        Calculate rewards based on fainted Pokémon, remaining HP, and victories.
        """
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.
        """
        if seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        
        # Call the parent class's reset method
        initial_obs = super().reset()

        # Embed the battle into a numerical representation
        observation = self.embed_battle(initial_obs)
        return np.array(observation, dtype=np.float32), {}  # Conform to Gymnasium's API

    def step(self, action):
        """
        Takes an action in the environment and returns the result.
        """
        # Call the parent class's step method
        battle, reward, done, info = super().step(action)

        # Embed the battle into a numerical representation
        observation = self.embed_battle(battle)
        return np.array(observation, dtype=np.float32), reward, done, info

# Training Parameters
NB_TRAINING_STEPS = 20_000
TEST_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)

if __name__ == "__main__":
    opponent = RandomPlayer()
    env_player = SimpleRLPlayer(opponent=opponent)
    env_player = Monitor(env_player)
    env_player = DummyVecEnv([lambda: env_player])

    model = A2C("MlpPolicy", env_player, verbose=1)
    model.learn(total_timesteps=NB_TRAINING_STEPS)

    # Testing the environment
    finished_episodes = 0
    env_player.reset_battles()
    obs, info = env_player.reset()

    while finished_episodes < TEST_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_player.step(action)
        if terminated or truncated:
            finished_episodes += 1
            obs, info = env_player.reset()

    print(f"Won {env_player.n_won_battles} battles against {env_player._opponent}")
