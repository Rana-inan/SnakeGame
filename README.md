Overview
A Snake game agent trained with Deep Q-Learning (DQN) using a CNN model.
The game is implemented with Pygame on a 10x10 grid and the agent learns to collect apples
while avoiding collisions.

What’s inside
• environment.py — Snake game environment
• brain.py — CNN Model (Brain)
• train.py — Training (DQN + replay memory)
• test.py — Play/Test using a saved model

How it works (short)
• The environment renders a 10x10 grid. Snake cells are marked as 0.5, apple cell is 1.
• The agent has 4 actions: 0=up, 1=down, 2=right, 3=left.
• A CNN takes stacked frames (nLastStates=4) as input and outputs Q-values for 4 actions.

Requirements
• Python 3.9+ (recommended)
• pygame, numpy
• tensorflow / keras
• (optional) matplotlib (training plot)

Install
pip install numpy pygame matplotlib tensorflow keras

Run the game manually (optional)
You can run the environment with keyboard control:
• Space: start/pause
• Arrow keys: move
python environment.py
Train the agent
python train.py

Notes about model files
This repository may not include *.h5 model files to keep it lightweight.
After training, a model file will be saved (e.g., model3.h5 by default).

Test / watch the trained agent play
test.py loads a model from:
• filepathToOpen = 'model.h5' (default)
If your saved model name is different (e.g., model3.h5), edit test.py and set:
filepathToOpen = 'model3.h5'
python test.py
Files
• train.py — training loop (epsilon-greedy, replay memory, CNN training)
• test.py — loads a saved .h5 model and plays automatically
• brain.py — CNN architecture (Conv2D → Pool → Conv2D → Dense)
• environment.py — grid rules, rewards, rendering, reset/step
• RedApple.png, Untitled design2.png — sprite assets used by the environment
Credits / Note
This project is based on a lecture/demo implementation of a CNN-DQN Snake agent.
