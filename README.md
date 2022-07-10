# Project 1: Navigation
by Luca Schweri, 2022

## Project Details

### Environment

In this environment, two agents control rackets to bounce a ball over a net. The goal is that the two players to collaborate to play the ball over the net.

### Rewards

- **+0.1**: If a player hits the ball over the net
- **-0.01**: If the ball hits the ground or a player hits the ball out of bounds.

### Actions

Each action is a vector with two numbers, corresponding to moving and jumping. Every entry in the action vector should be a number between -1 and 1.

### States

The observation space consists of 8 variables corresponding to position and velocity of the ball and rackets.

### Solved

The environment is solved as soon as the agent is able to get an average score higher than 0.5 over 100 consecutive episodes. The score of one episode is the maximum score of the two players.


## Getting Started

To set up your python environment use the following commands:

```
conda create --name udacity_rl python=3.6
conda activate udacity_rl

pip install -r requirements.txt
conda install -n udacity_rl pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```

Download the Unity environment and change the path to the environment in the first code cell of the jupyter notebook [navigation.ipynb](navigation.ipynb):

Tennis Environment:
- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Instructions

To train or test a new approach, first modify the [config.json](config.json) file and then open the jupyter notebook [navigation.ipynb](navigation.ipynb) by using the command
```
jupyter notebook
```

In the notebook you find three sections (which are better described in the jupyter notebook):
- **Trainig**: Can be used to train an agent.
- **Test**: Can be used to test an already trained agent.

Note that you might need to change the location of the Unity environment in the first code cell.