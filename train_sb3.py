import datetime
import argparse

import numpy as np
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from agents.sub_optimal_agent import SubOptimalAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLModelAgent, RLTrainingAgent
from agents.value_agent import ValueAgent

np.random.seed(123)
ENV_NAME = 'ForSale-v0'
gym.envs.registration.register(id=ENV_NAME, entry_point='env:ForSale')

class ModelTrainer:
    def __init__(self, num_players=3, agents='ss', seed=123, steps=1e6, models=[]):
        self.num_players = num_players
        self.agents = agents
        self.seed = seed
        self.steps = steps
        self.model_name = None
        
        self.players = []
        player_num = 0
        m_count = 0
        for agent in agents:
            if agent == 's':
                self.players.append(SubOptimalAgent(player_num))
            elif agent == 'r':
                self.players.append(RandomAgent(player_num))
            elif agent == 'v':
                self.players.append(ValueAgent(player_num))
            elif agent == 'm':
                self.players.append(RLModelAgent(player_num, algo='ppo_mask', path=models[m_count]))
                m_count += 1
            player_num += 1
        self.players.append(RLTrainingAgent(player_num))

        self.env_kwargs = {
            'players': self.players,
            'live_player': player_num
        }

    def run(self, algo='ppo_mask'):
        time_str = datetime.datetime.now().strftime('%y.%m.%d_%H.%M.%S')
        self.model_name = '{}_{}_{}_{}_{}'.format(algo, str(self.num_players), self.agents, str(self.steps), time_str)

        if algo == 'ppo':
            self.ppo_model()
        elif algo == 'dqn':
            self.dqn_model()
        elif algo == 'ppo_mask':
            self.ppo_mask_model()

    # PPO
    def ppo_model(self):
        env = make_vec_env(ENV_NAME, env_kwargs=self.env_kwargs)
        env.env_method('reset', seed=self.seed)

        model = PPO('MlpPolicy', env, seed=self.seed, verbose=0, tensorboard_log='./tensorboard/ppo/' + self.model_name)
        model.learn(total_timesteps=self.steps)
        model.save('./models/ppo/' + self.model_name)

    # DQN 
    def dqn_model(self):
        env = gym.make(ENV_NAME, **self.env_kwargs)
        env.reset(seed=self.seed)

        model = DQN('MlpPolicy', env, seed=self.seed, verbose=0, tensorboard_log='./tensorboard/dqn/' + self.model_name)
        model.learn(total_timesteps=self.steps)
        model.save('./models/dqn/' + self.model_name)

    # PPO maskable
    def ppo_mask_model(self):
        def mask_fn(env: gym.Env) -> np.ndarray:
            return env.get_action_mask()

        env = gym.make(ENV_NAME, **self.env_kwargs)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=self.seed)

        model = MaskablePPO(MaskableActorCriticPolicy, env, seed=self.seed, verbose=0, tensorboard_log='./tensorboard/ppo_mask/' + self.model_name)
        model.learn(total_timesteps=self.steps)
        model.save('./models/ppo_mask/' + self.model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model trainer arguments')
    parser.add_argument('--num_players', type=int, default=3)
    parser.add_argument('--agents', default='ss')
    parser.add_argument('--steps', type=int, default=2e6)
    parser.add_argument('--models', nargs='*', default=[])

    args = parser.parse_args().__dict__
    num_players = args['num_players']
    agents = args['agents']
    steps = args['steps']
    models = args['models']

    if len(agents) != num_players - 1:
        raise Exception('must have agents equal to num_players - 1')

    if agents.count('m') != len(models):
        raise Exception('must have models equal to amount of model agents')

    mt = ModelTrainer(num_players=num_players, agents=agents, steps=steps, models=models)
    mt.run()
