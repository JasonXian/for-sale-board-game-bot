# initial implementation using keras-rl that no longer works

import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory

from agents.sub_optimal_agent import SubOptimalAgent
from agents.random_agent import RandomAgent
from agents.rl_agent import RLTrainingAgent

np.random.seed(123)
ENV_NAME = 'ForSale-v0'
gym.envs.registration.register(id=ENV_NAME, entry_point='env:ForSale')
env = gym.make(ENV_NAME)

env.add_player(SubOptimalAgent())
env.add_player(SubOptimalAgent())
env.add_player(RLTrainingAgent(), is_live_player=True)
env.reset(seed=123, return_info=True)

nb_actions = env.action_space.n
model = Sequential()
model.add(Flatten(input_shape=(1, len(env.observation))))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_actions, activation='linear'))

# model.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
# After training is done, we save the final weights.

memory = SequentialMemory(limit=50000, window_length=1)
policy = MaxBoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
h = dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
 
# testing
h = dqn.test(env, nb_episodes=50, visualize=True)
hist = h.history
winrate = sum(x >= 1 for x in hist['episode_reward']) / len(hist['episode_reward'])
print('winrate {}'.format(winrate))
