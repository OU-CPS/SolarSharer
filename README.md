# SolarSharer

SolarSharer is a decentralized, multi-agent environment for simulating peer-to-peer solar energy trading across multiple houses, with the goal of making energy distribution fairer, more stable, and more sustainable. Built on an OpenAI Gym interface, it models the dynamic, decentralized nature of Virtual Power Plants (VPPs) and supports reward shaping, step/observation spaces, and action spaces tailored for P2P energy markets.

We provide a main environment file, `solar_sharer_env.py`, which contains the core logic—from the reward function to the reset and step methods. The accompanying MADDPG folder and experiments scripts illustrate how to train using Multi-Agent Deep Deterministic Policy Gradient, with pointers on hyperparameter tuning. In addition, the codebase shows how to adapt other RL algorithms such as PPO, Policy Gradient, Independent DQN, or even Round Robin scheduling. By following these steps, you can integrate any multi-agent RL method and extend the environment to tackle more sophisticated, nonstationary scenerios in VPP systems.

## 1. Overview
#### 1.1. The SolarSharer Environment
Location: `solar_sharer_env.py`

Purpose: Simulates a group of residential solar prosumers and consumers, tracking each agent’s demand, solar generation, and potential trades.

Key Methods:

`__init__()`: Loads and preprocesses real-world energy data, sets up spaces, reward parameters, etc.

`reset()`: Resets the environment to a new day’s data (or random day) and returns the initial observation for each agent.

`step(actions)`:

Validates continuous actions (buy/sell from grid or peers).

Calculates how much each agent imports or exports.

Applies a reward function balancing cost savings, fairness, and more.

Returns (new_observations, rewards, done, info) just like any other Gym environment.

Reward Function: A multi-objective formulation encouraging minimal grid reliance, active peer-to-peer participation, and fairness (via Jain’s Index).

1.2. Multi-Agent RL
We provide 4 training scripts that use this environment:

`train.py`: Uses a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.

`train_ppo.py`: Uses a Proximal Policy Optimization (PPO) variant adapted for multi-agent settings.

You can switch algorithms or create new ones as long as you follow the Gym “Env” interface.

3. Installation & Environment Setup
Clone or copy the repository:

bash

```bash

python3 -m venv venv
source venv/bin/activate

```
Install required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

(Optional) GPU Setup:

If you want to use CUDA, ensure PyTorch is installed with GPU support and that you have the necessary drivers.

If you see “No GPU detected” logs, it’ll run on CPU.


