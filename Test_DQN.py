import numpy as np
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gym.spaces import Space, Box
from poke_env.player import Gen8EnvSinglePlayer, ObsType
from gym.utils.env_checker import check_env
from poke_env.player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tabulate import tabulate
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

class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

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

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

async def main():
    opponent = RandomPlayer(battle_format="gen1randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen1randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    opponent = RandomPlayer(battle_format="gen1randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen1randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen1randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen1randombattle", opponent=opponent, start_challenging=True
    )

    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    model = QNetwork(np.prod(input_shape), n_action)

    optimizer = optim.Adam(model.parameters(), lr=0.00025)

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

    # Training
    for _ in range(10000):
        batch = memory.sample(1)
        states = torch.tensor([b.state for b in batch])
        actions = torch.tensor([b.action for b in batch])
        rewards = torch.tensor([b.reward for b in batch])
        next_states = torch.tensor([b.next_state for b in batch])
        dones = torch.tensor([b.done for b in batch])

        q_values = model(states)
        q_values_next = model(next_states).max(1)[0].detach()

        target_q_values = rewards + (1 - dones) * 0.5 * q_values_next

        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(selected_q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    for _ in range(100):
        dqn.test(eval_env, nb_episodes=1, visualize=False)
    eval_env.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
