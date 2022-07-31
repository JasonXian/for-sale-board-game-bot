import random
from env import Player, Action, Stage

# agent performs random legal actions
class RandomAgent(Player):
    def __init__(self, player_num):
        super().__init__(player_num)

    def action(self, info):
        action_mask = info['action_mask']
        actions = []
        for i, is_action_valid in enumerate(action_mask):
            if is_action_valid:
                actions.append(Action.TAKE + i)

        return random.choice(actions)
    