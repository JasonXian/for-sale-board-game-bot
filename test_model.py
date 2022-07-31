import argparse

import gym
from sb3_contrib import MaskablePPO
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLTrainingAgent, RLModelAgent
from agents.sub_optimal_agent import SubOptimalAgent
from agents.value_agent import ValueAgent

import env

# test model against real player to analyze performance
ENV_NAME = 'ForSale-v0'
gym.envs.registration.register(id=ENV_NAME, entry_point='env:ForSale')

players = [RLTrainingAgent(0), HumanAgent(1)]
env_kwargs = {
    'players': players,
    'live_player': 0,
    'render_mode': 'text'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model trainer arguments')
    parser.add_argument('--num_players', type=int, default=3)
    parser.add_argument('--agents', default='s')
    parser.add_argument('--models', nargs='*', default=[])

    args = parser.parse_args().__dict__
    num_players = args['num_players']
    agents = args['agents']
    models = args['models']
    
    if len(models) == 0:
        raise Exception('requires model path name')

    if len(agents) != num_players - 2:
        raise Exception('must have agents equal to num_players - 2')

    if agents.count('m') + 1 != len(models):
        raise Exception('must have models equal to amount of model agents + 1')

    # test model is player 0, taken from first path of models array
    # human agent is player 1
    player_num = 2
    m_count = 1
    for agent in agents:
        if agent == 's':
            env_kwargs['players'].append(SubOptimalAgent(player_num))
        elif agent == 'r':
            env_kwargs['players'].append(RandomAgent(player_num))
        elif agent == 'v':
            env_kwargs['players'].append(ValueAgent(player_num))
        elif agent == 'm':
            env_kwargs['players'].append(RLModelAgent(player_num, path=models[m_count]))
            m_count += 1
        player_num += 1

    env = gym.make(ENV_NAME, **env_kwargs)
    model = MaskablePPO.load(models[0])
    obs = env.reset()
    while True:
        action_mask = env.get_action_mask()
        action, _ = model.predict(obs, action_masks=action_mask)
        obs, rewards, done, info = env.step(action)

        if done:
            break
