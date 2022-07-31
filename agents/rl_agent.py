from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO

from env import Player

# empty agent to stand in for the training network
class RLTrainingAgent(Player):
    def __init__(self, player_num):
        super().__init__(player_num)

# agent that loads and uses an existing model
class RLModelAgent(Player):
    def __init__(self, player_num, algo='ppo_mask', path=None):
        super().__init__(player_num)
        if not path:
            raise Exception('need valid path model')

        self.algo = algo
        self.model = None
        if algo == 'ppo':
            self.model = PPO.load(path)
        elif algo == 'dqn':
            self.model = DQN.load(path)
        elif algo == 'ppo_mask':
            self.model = MaskablePPO.load(path)

    def action(self, info):
        action = None
        if self.algo == 'ppo_mask':
            action, _ = self.model.predict(info['observation'], action_masks=info['action_mask'])
        else:
            action, _ = self.model.predict(info['observation'])

        return action
