# SolarSharer

This project provides a SolarSharer environment that simulates peer-to-peer solar energy trading among multiple houses. SolarSharer is a decentralized P2P energy trading system. We implement SolarSharer using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) framework and design both the environment and
reward function to capture the dynamic and decentralized characteristics of energy trading the VPP scenarios.
We have the main environment file solar_sharer_env.py which had everything reward function, step etc. Then as mentioned in our paper we demonstrate training using other alogorithm as well. MADDPG, PPO, Policy Gradient, and Independent DQN, you can integrate any multi-agent RL algorithm (e.g., MAPPO, QMIX, etc.) by following these steps.

Below we explain step by step how you can use this and more over extend this environment to implement more efficient algorithms for true multiagent/nonstationary environment.
