import asyncio
import numpy as np
import torch
from stable_baselines3 import DQN
from gymnasium.spaces import Box
from poke_env.player import Gen9EnvSinglePlayer, Player
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from poke_env.data import GenData


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


class DQNPlayer(Gen9EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_data = GenData.from_gen(9)

    def embed_battle(self, battle):
        """Battle state embedding method"""
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
        """Define observation space"""
        low = np.array([-1.0] * 4 + [0.0] * 4 + [0.0, 0.0], dtype=np.float32)
        high = np.array([1.0] * 4 + [4.0] * 4 + [1.0, 1.0], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def calc_reward(self, last_state, current_state):
        """Calculate reward based on battle state"""
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        initial_obs = super().reset(seed=seed, options=options)
        observation = self.embed_battle(initial_obs)
        return observation, {}

    def step(self, action):
        """Take a step in the environment."""
        step_output = super().step(action)

        if len(step_output) == 4:  # Old convention: (observation, reward, done, info)
            battle, reward, done, info = step_output
            terminated, truncated = done, False
        elif len(step_output) == 5:  # New convention: (observation, reward, terminated, truncated, info)
            battle, reward, terminated, truncated, info = step_output
        else:
            raise ValueError(f"Unexpected step output: {step_output}")

        observation = self.embed_battle(battle)
        return observation, reward, terminated, truncated, info


async def main():
    np.random.seed(42)
    torch.manual_seed(42)

    opponent = MaxDamagePlayer()
    envOG_player = DQNPlayer(opponent=opponent)
    env_player = Monitor(envOG_player)
    env_player = DummyVecEnv([lambda: env_player])

    model = DQN(
        "MlpPolicy",
        env_player,
        verbose=1,
        seed=42,
        learning_rate=1e-3,
        buffer_size=100000,
        batch_size=64,
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
        target_update_interval=1000,
        #policy_kwargs=dict(double_q=True)  # Explicitly enable Double DQN
    )
    model.learn(total_timesteps=100_000)
    print("#### Training against Max Damage Bot #####")
    print("Won", envOG_player.n_won_battles, "total battle", envOG_player.n_finished_battles)


if __name__ == "__main__":
    asyncio.run(main())
