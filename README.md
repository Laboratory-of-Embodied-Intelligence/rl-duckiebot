![Docker Image CI](https://github.com/Laboratory-of-Embodied-Intelligence/rl-duckiebot/workflows/Docker%20Image%20CI/badge.svg)

[![DeepSource](https://static.deepsource.io/deepsource-badge-dark-mini.svg)](https://deepsource.io/gh/Laboratory-of-Embodied-Intelligence/rl-duckiebot/?ref=repository-badge)

# Outline
Work in progress repository, that contains various rl algorithms and classic control theory benchmarked in duckietown
Current algortihms will be:
1. Classic PID.
2. Deep Deterministic Policy Gradient Algorithm and it's modifications. Currently implemented DDPG with VAE taken from https://github.com/araffin/learning-to-drive-in-5-minutes


# Installation
1. Install https://github.com/Laboratory-of-Embodied-Intelligence/gym-duckietown with instructions in readme.
2. ```conda env create --name MY_ENV_NAME -f environment.yml```
3. ```cd rl && python3 stable_train.py``` will train DDPG+VAE, ```cd rl && python3 train.py``` will train vanilla version of DDPG taken from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code

# Training parameters
There are a lot of hyperparameters to choose, all list is written in stable_train.py, by tweaking them you can drastically change behaviour of the model.
