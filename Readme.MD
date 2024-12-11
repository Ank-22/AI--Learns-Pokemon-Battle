# Pokémon Battles with Reinforcement Learning

This project explores the application of advanced reinforcement learning (RL) algorithms to the strategic domain of Pokémon battles, incorporating the latest mechanics such as Dynamax and Terastalization. The goal is to develop and evaluate AI agents capable of competing in turn-based strategy games by leveraging sophisticated RL techniques.

## Overview

Pokémon battles are highly complex environments, characterized by large state spaces, hidden information, and dynamic strategies. This project formulates Pokémon battles as a Markov Decision Process (MDP) and applies three RL algorithms to train autonomous agents:

- **Proximal Policy Optimization (PPO)**  
- **Deep Q-Networks (DQN)**  
- **Asynchronous Advantage Actor-Critic (A3C)**  

The agents are trained and tested in simulated environments using the [Pokémon Showdown](https://pokemonshowdown.com/) platform and the `poke-env` library.

## Key Features

- **Dynamic Battle Mechanics**: Support for Generations 8 (Dynamax) and 9 (Terastalization).  
- **Algorithm Comparisons**: Performance analysis of PPO, DQN, and A3C in simple and complex battle environments.  
- **Custom Rule-Based Bots**: Development of rule-based opponents (e.g., Max Damage, Type Advantage) to benchmark RL agents.  
- **Reward Optimization**: Design of adaptive reward structures to align AI strategies with successful battle outcomes.  

## Results

- **DQN**: Achieved a win rate of **77.5%** in Generation 4 battles but struggled with advanced mechanics like Terastalization.  
- **PPO**: Demonstrated moderate adaptability, with a peak win rate of **38.69%** in Generation 8 battles against strong opponents.  
- **A3C**: Showed promise in dynamic environments, steadily improving rewards and performance with complex mechanics.  

## Goals and Applications

This project advances the application of RL in turn-based strategy games, providing insights into multi-agent systems, game AI, and dynamic decision-making. Future work includes optimizing reward functions, enhancing scalability, and exploring generalization across other strategic domains.

## How to run the project
### 1] Setting up the virtual enviorement
```
python -m venv venv
```
### 2] Activating the virtual enviorement
For windows:
```
venv\Scripts\activate
```

For Mac/Linux:
```
source venv/bin/activate
```
### 3] Installing dependencies:
```
pip install -r requirements.txt
```
### 4] Verify installation:
```
pip list
```

### 5] Run the Pokemon Showdown Sever
```
cd to pokemon-showdown
```
1) Install *NPM* 
2)


```
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

### 6] Run the Pokemon RL Script
```
python PokePPO.py
```
or 
```
python PokeA3C.py
```
or 
```
python PokeTestPoke.py
```

## Acknowledgements

This project is a follow-up of a group project from an artifical intelligence class at [Ecole Polytechnique](https://www.polytechnique.edu/).

You can find the original repository [here](https://github.com/hsahovic/inf581-project). It is partially inspired by the [showdown-battle-bot project](https://github.com/Synedh/showdown-battle-bot). Of course, none of these would have been possible without [Pokemon Showdown](https://github.com/Zarel/Pokemon-Showdown).

Team data comes from [Smogon forums' RMT section](https://www.smogon.com/).

## Thanks to Poke-Env:

For making a python warrper and making example for OpenAI Gym.

@misc{poke_env,
    author       = {Haris Sahovic},
    title        = {Poke-env: pokemon AI in python},
    url          = {https://github.com/hsahovic/poke-env}
}

