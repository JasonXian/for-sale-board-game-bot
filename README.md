# For Sale Board Game Bot

Simple open api gym environment and for training a RL model using stable baselines to play the board game for sale.

## Usage

Install reqs: `pip install -r requirements.txt`

Train a model: `python train_sb3.py --num_players <n> --agents <a> --steps <s> --models <m>`

- n = num players from 3 to 6
- a = cpu agent types in str form, e.g. "srvm" is a 4 player game with a agents of type: suboptimal, random, value and model
- s = num steps to run training
- m = space seperated model file paths

Interactively test a model: `python test_model.py --num_players <n> --agents <a> --models <m>`

- n = num players from 3 to 6 (the first two players are defaulted for the model to be tested and a human player)
- a = cpu agent types in str form, e.g. "srvm" is a 6 player game with a agents of type: suboptimal, random, value and model
- m = space seperated model file paths, at least one path required

e.g. python test_model.py --num_players 4 --agents vv --models ./models/ppo_mask/ppo_mask_4_rsr_2000000_1657471736.8542898
