import os
import asyncio
from stable_baselines3 import PPO
from poke_env.player import Player
from poke_env.environment.battle import Battle
from PokePPO import PPOBot  # Replace with your PPO bot implementation
from rp import RandomPlayer  # Example opponent, can be replaced with others

# Define the path to your saved PPO model
model_path = 'model/PPO/lr3_ns_2048_100K_Gen4.pt'

# Custom PPOBot class to integrate the loaded model
class MyPPOBot(PPOBot):
    def __init__(self, model, opponent, battle_format="gen4randombattle"):
        # Pass the opponent to the parent class
        super().__init__(opponent=opponent, battle_format=battle_format)
        self.model = model  # Load the PPO model into the bot

    async def _send_challenges(self, opponent_username, n_challenges=1):
        """
        Custom method to send challenges to a human player.
        Args:
            opponent_username: The username of the human opponent.
            n_challenges: Number of challenges to send.
        """
        print(f"Sending {n_challenges} challenge(s) to {opponent_username}...")
        await self.send_challenges(opponent_username, n_challenges=n_challenges)
        print(f"Challenges sent to {opponent_username}!")

    def choose_move(self, battle: Battle):
        """
        Override the choose_move method to make a decision based on the PPO model.
        """
        state = self.embed_battle(battle)
        action, _ = self.model.predict(state)
        return self._action_to_move(action, battle)

# Check if the model exists and load it
if os.path.exists(model_path):
    print(f"Loading PPO model from {model_path}")
    ppo_model = PPO.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Replace this with your Pokémon Showdown username
your_username = "zxcvbnm22"

# Define an opponent for the PPOBot
opponent = RandomPlayer(battle_format="gen4randombattle")

# Initialize PPOBot with the loaded model and the opponent
ppo_bot = MyPPOBot(model=ppo_model, opponent=opponent, battle_format="gen4randombattle")

async def main():
    # Send a challenge to the human player
    await ppo_bot._send_challenges(opponent_username=your_username, n_challenges=1)

    # Wait for the battle to complete
    while ppo_bot.n_finished_battles < 1:
        await asyncio.sleep(1)

    print(f"PPOBot finished {ppo_bot.n_finished_battles} battles!")
    print(f"PPOBot won {ppo_bot.n_won_battles} battle(s).")

# Run the event loop
asyncio.run(main())
