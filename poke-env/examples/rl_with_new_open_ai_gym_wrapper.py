import asyncio

import numpy as np
from gymnasium.spaces import Box, Space
from gymnasium.utils.env_checker import check_env
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    Gen8EnvSinglePlayer,
    MaxBasePowerPlayer,
    ObsType,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    background_cross_evaluate,
    background_evaluate_player,
)

from typing import Dict

type_chart: Dict[str, Dict[str, float]] = {
    'Normal': {'Normal': 1.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 0.5, 'Ghost': 0.0, 'Dragon': 1.0, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 1.0},
    'Fire': {'Normal': 1.0, 'Fire': 0.5, 'Water': 0.5, 'Electric': 1.0, 'Grass': 2.0, 'Ice': 2.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 2.0, 'Rock': 0.5, 'Ghost': 1.0, 'Dragon': 0.5, 'Dark': 1.0, 'Steel': 2.0, 'Fairy': 1.0},
    'Water': {'Normal': 1.0, 'Fire': 2.0, 'Water': 0.5, 'Electric': 1.0, 'Grass': 0.5, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 2.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 2.0, 'Ghost': 1.0, 'Dragon': 0.5, 'Dark': 1.0, 'Steel': 1.0, 'Fairy': 1.0},
    'Electric': {'Normal': 1.0, 'Fire': 1.0, 'Water': 2.0, 'Electric': 0.5, 'Grass': 0.5, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 0.0, 'Flying': 2.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 1.0, 'Dragon': 0.5, 'Dark': 1.0, 'Steel': 1.0, 'Fairy': 1.0},
    'Grass': {'Normal': 1.0, 'Fire': 0.5, 'Water': 2.0, 'Electric': 1.0, 'Grass': 0.5, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 0.5, 'Ground': 2.0, 'Flying': 0.5, 'Psychic': 1.0, 'Bug': 0.5, 'Rock': 2.0, 'Ghost': 1.0, 'Dragon': 0.5, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 1.0},
    'Ice': {'Normal': 1.0, 'Fire': 0.5, 'Water': 0.5, 'Electric': 1.0, 'Grass': 2.0, 'Ice': 0.5, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 2.0, 'Flying': 2.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 1.0, 'Dragon': 2.0, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 1.0},
    'Fighting': {'Normal': 2.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 2.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2.0, 'Ghost': 0.0, 'Dragon': 1.0, 'Dark': 2.0, 'Steel': 2.0, 'Fairy': 0.5},
    'Poison': {'Normal': 1.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 2.0, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 0.5, 'Ground': 0.5, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 0.5, 'Ghost': 0.5, 'Dragon': 1.0, 'Dark': 1.0, 'Steel': 0.0, 'Fairy': 2.0},
    'Ground': {'Normal': 1.0, 'Fire': 2.0, 'Water': 1.0, 'Electric': 2.0, 'Grass': 0.5, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 2.0, 'Ground': 1.0, 'Flying': 0.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 2.0, 'Ghost': 1.0, 'Dragon': 1.0, 'Dark': 1.0, 'Steel': 2.0, 'Fairy': 1.0},
    'Flying': {'Normal': 1.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 0.5, 'Grass': 2.0, 'Ice': 1.0, 'Fighting': 2.0, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 2.0, 'Rock': 0.5, 'Ghost': 1.0, 'Dragon': 1.0, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 1.0},
    'Psychic': {'Normal': 1.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 1.0, 'Fighting': 2.0, 'Poison': 2.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 0.5, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 1.0, 'Dragon': 1.0, 'Dark': 0.0, 'Steel': 0.5, 'Fairy': 1.0},
    'Bug': {'Normal': 1.0, 'Fire': 0.5, 'Water': 1.0, 'Electric': 1.0, 'Grass': 2.0, 'Ice': 1.0, 'Fighting': 0.5, 'Poison': 0.5, 'Ground': 1.0, 'Flying': 0.5, 'Psychic': 2.0, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 0.5, 'Dragon': 1.0, 'Dark': 2.0, 'Steel': 0.5, 'Fairy': 0.5},
    'Rock': {'Normal': 1.0, 'Fire': 2.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 2.0, 'Fighting': 0.5, 'Poison': 1.0, 'Ground': 0.5, 'Flying': 2.0, 'Psychic': 1.0, 'Bug': 2.0, 'Rock': 1.0, 'Ghost': 1.0, 'Dragon': 1.0, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 1.0},
    'Ghost': {'Normal': 0.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 1.0, 'Fighting': 0.0, 'Poison': 0.5, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 0.5, 'Rock': 1.0, 'Ghost': 2.0, 'Dragon': 1.0, 'Dark': 2.0, 'Steel': 1.0, 'Fairy': 1.0},
    'Dragon': {'Normal': 1.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 1.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 1.0, 'Dragon': 2.0, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 0.0},
    'Dark': {'Normal': 1.0, 'Fire': 1.0, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 1.0, 'Fighting': 0.5, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 2.0, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 0.5, 'Dragon': 1.0, 'Dark': 0.5, 'Steel': 1.0, 'Fairy': 0.5},
    'Steel': {'Normal': 1.0, 'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Grass': 1.0, 'Ice': 2.0, 'Fighting': 1.0, 'Poison': 1.0, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 2.0, 'Ghost': 1.0, 'Dragon': 1.0, 'Dark': 1.0, 'Steel': 0.5, 'Fairy': 0.5},
    'Fairy': {'Normal': 1.0, 'Fire': 0.5, 'Water': 1.0, 'Electric': 1.0, 'Grass': 1.0, 'Ice': 1.0, 'Fighting': 2.0, 'Poison': 0.5, 'Ground': 1.0, 'Flying': 1.0, 'Psychic': 1.0, 'Bug': 1.0, 'Rock': 1.0, 'Ghost': 1.0, 'Dragon': 2.0, 'Dark': 2.0, 'Steel': 2.0, 'Fairy': 1.0}
}



class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training the model
    dqn.fit(train_env, nb_steps=10000)
    train_env.close()

    # Evaluating the model
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        eval_env.agent, n_challenges, placement_battles
    )
    dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
