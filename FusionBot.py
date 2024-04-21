import asyncio
import time
import numpy as np
from poke_env.player import Player, RandomPlayer
from rp import MaxDamagePlayer
from TypeAdvantageBot import TypeAdvantagePlayer

class FusionPlayer(Player):
    def choose_move(self, battle):
        try:
            picker = np.random(0,2)
        except:
            picker =0
        if(picker == 0):
            if(battle.available_moves):
                moveTypePresent = False
                for i, move in enumerate(battle.available_moves):
                
                    moves_base_power = -np.zeros(4)
                    moves_dmg_multiplier = np.ones(4)
                    moves_dmg = np.zeros(4)
                    moves_base_power[i] = (
                        move.base_power / 100
                    )  # Simple rescaling to facilitate learning
                    # if move.type:
                    #     moves_dmg_multiplier[i] = np.random(0,4)
                    if move.type:
                        moveTypePresent = True
                        moves_dmg_multiplier[i] = move.type.damage_multiplier(
                            battle.opponent_active_pokemon.type_1,
                            battle.opponent_active_pokemon.type_2,
                        )
                        moves_dmg[i] = moves_base_power[i]* moves_dmg_multiplier[i]
                    else:
                        moves_dmg[i] = moves_base_power[i]
                if(moveTypePresent == True):
                    try:
                        best_move = battle.available_moves[np.where(moves_dmg == max(moves_dmg))[0][0]]
                        
                    except IndexError:
                            print("something went wrong will picking the move for type advantage player")
                    return self.create_order(best_move)
                # If no attack is available, a random switch will be made
                else:
                    best_move = max(battle.available_moves, key=lambda move: move.base_power)
                    return self.create_order(best_move)
            else:
                return self.choose_random_move(battle)
        elif (picker == 1):
              # If the player can attack, it will
            if battle.available_moves:
                # Finds the best move among available ones
                best_move = max(battle.available_moves, key=lambda move: move.base_power)
                return self.create_order(best_move)

            # If no attack is available, a random switch will be made
            else:
                return self.choose_random_move(battle)
        else:
            return self.choose_random_move(battle)

async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(
        battle_format="gen1randombattle",
    )
    Type_Advantage_player = TypeAdvantagePlayer(
        battle_format="gen1randombattle",
    )
    max_damager_player = MaxDamagePlayer(
         battle_format="gen1randombattle",
    )
    Fusion_player = FusionPlayer(
         battle_format="gen1randombattle",
    )

    # Now, let's evaluate our player
    await Type_Advantage_player.battle_against(max_damager_player, n_battles=500)
    await Type_Advantage_player.battle_against(random_player, n_battles=500)
    await Type_Advantage_player.battle_against(Fusion_player, n_battles=500)
    await random_player.battle_against(max_damager_player, n_battles=500)
    await random_player.battle_against(Fusion_player, n_battles=500)
    await Fusion_player.battle_against(max_damager_player, n_battles=500)



    print(
        "Type Advantage player won %d / 3000 battles[this took %f seconds]"
        % (
            Type_Advantage_player.n_won_battles, time.time() - start
        )
    )
    print(
        "Random player won %d / 3000 battles[this took %f seconds]"
        % (
            random_player.n_won_battles, time.time() - start
        )
    )

    print(
        "Max Damage player won %d / 3000 battles[this took %f seconds]"
        % (
            max_damager_player.n_won_battles, time.time() - start
        )
    )
    print(
        "Fusion Player won %d / 3000 battles[this took %f seconds]"
        % (
            Fusion_player.n_won_battles, time.time() - start
        )
    )



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
