# SolarSharer

This project provides a SolarSharer environment that simulates peer-to-peer solar energy trading among multiple houses. SolarSharer is a decentralized P2P energy trading system. We implement SolarSharer using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) framework and design both the environment and
reward function to capture the dynamic and decentralized characteristics of energy trading the VPP scenarios.
We have the main environment file solar_sharer_env.py which had everything reward function, step etc. Then as mentioned in our paper we demonstrate training using other alogorithm as well. MADDPG, PPO, Policy Gradient, and Independent DQN, you can integrate any multi-agent RL algorithm (e.g., MAPPO, QMIX, etc.) by following these steps.

Below we explain step by step how you can use this and more over extend this environment to implement more efficient algorithms for true multiagent/nonstationary environment.

1. Overview
2. 1.1. The SolarSharer Environment
Location: solar_sharer_env.py

Purpose: Simulates a group of residential solar prosumers and consumers, tracking each agent’s demand, solar generation, and potential trades.

Key Methods:

__init__(): Loads and preprocesses real-world energy data, sets up spaces, reward parameters, etc.

reset(): Resets the environment to a new day’s data (or random day) and returns the initial observation for each agent.

step(actions):

Validates continuous actions (buy/sell from grid or peers).

Calculates how much each agent imports or exports.

Applies a reward function balancing cost savings, fairness, and more.

Returns (new_observations, rewards, done, info) just like any other Gym environment.

Reward Function: A multi-objective formulation encouraging minimal grid reliance, active peer-to-peer participation, and fairness (via Jain’s Index).

1.2. Multi-Agent RL
We provide two training scripts that use this environment:

train.py: Uses a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.

train_ppo.py: Uses a Proximal Policy Optimization (PPO) variant adapted for multi-agent settings.

You can switch algorithms or create new ones as long as you follow the Gym “Env” interface.

3. Installation & Environment Setup
Clone or copy the repository:

bash

git clone https://github.com/YourUserName/SolarSharer.git
cd SolarSharer
Create a virtual environment (optional but recommended):

bash

python3 -m venv venv
source venv/bin/activate
Install required packages:

bash

pip install --upgrade pip
pip install -r requirements.txt
(Adjust version pins in requirements.txt if needed.)

(Optional) GPU Setup:

If you want to use CUDA, ensure PyTorch is installed with GPU support and that you have the necessary drivers.

If you see “No GPU detected” logs, it’ll run on CPU.


