from enum import IntEnum
from math import ceil

import gym
import numpy as np

class Action (IntEnum):
    # either take or bid 1-2 coins more
    TAKE = 0
    BID_1 = 1
    BID_2 = 2
    # sell up to 8 cards depending on game size
    SELL_1 = 3
    SELL_2 = 4
    SELL_3 = 5
    SELL_4 = 6
    SELL_5 = 7
    SELL_6 = 8
    SELL_7 = 9
    SELL_8 = 10

class Stage (IntEnum):
    BUYING = 0
    SELLING = 1

# player represents state of agent in game
class Player ():
    def __init__(self, player_num):
        self.num = player_num
        self.coins = 0
        self.money = 0
        self.property = []
        self.is_bidding = True
        self.bid = 0
        self.sell_property = 0

    def __repr__(self):
        return 'P{} | $: {} B: {} M: {} H: {}'.format(self.num, self.coins, self.bid, self.money, self.property)

class ForSale (gym.Env):
    NUM_CARDS = {
        3: 24,
        4: 28,
        5: 30,
        6: 30
    }
    NUM_COINS = {
        3: 18,
        4: 18,
        5: 14,
        6: 14
    }
    NUM_COINS_BIG_MONEY = {
        3: 28,
        4: 21,
        5: 16,
        6: 14
    }

    # pass in index of live player
    def __init__(self, players, live_player=-1, render_mode='none'):
        self.render_mode = render_mode
        self.num_players = len(players)
        self.players = players
        self.live_player = live_player
        self.action_space = gym.spaces.Discrete(len(Action))
        self.reset(options={'init': True})

    def reset(self, return_info=False, seed=None, options=None):
        if self.num_players < 3 or self.num_players > 6:
            raise Exception('number of players must be 3-6')

        # initial 30 cards
        self.property_cards = np.arange(1, 31)
        self.money_cards = np.concatenate([[0, 0], np.arange(2, 16), np.arange(2, 16)])
        np.random.seed(seed)
        np.random.shuffle(self.property_cards)
        np.random.shuffle(self.money_cards)
        while len(self.property_cards) != self.NUM_CARDS[self.num_players]:
            self.property_cards = np.delete(self.property_cards, 0)
            self.money_cards = np.delete(self.money_cards, 0)

        # player defaults
        cards_per_player = self.NUM_CARDS[self.num_players] // self.num_players
        for player in self.players:
            player.coins = self.NUM_COINS[self.num_players]
            player.money = 0
            player.is_bidding = True
            player.bid = 0
            player.property = [-1] * cards_per_player
            player.sell_property = -1

        # set board and stage
        self.stage = Stage.BUYING
        self.num_buy_round = 0
        self.num_buy_property = 0
        self.board = self._get_next_cards()
        self.curr_player = np.random.randint(self.num_players)
        self.last_bid = 0
        self.num_sell_round = 0
        self.num_sell_property = 0
        self.is_game_over = False
        self.reward = 0

        # for render function
        self.action = None
        self.prev_player = None
        self.sell_round_str = None

        self._get_observation()
        self._get_info()
        
        if not (options and options['init']):
            if self.render_mode != 'none':
                self._render()
            # autoplay in case first player isn't the live player
            self._auto_play()

        if return_info:
            return self.observation, self.info

        return self.observation

    # live agent calls step, auto play all other players after executing step
    def step(self, action):
        self.reward = 0
        self._execute_action(action)
        self._auto_play()
        return (self.observation, float(self.reward), self.is_game_over, self.info)
        
    def _execute_action(self, action):
        self.action = action
        self.prev_player = self.curr_player

        if self.stage == Stage.BUYING:
            self._resolve_buy_action(action)
        else:
            self._resolve_sell_action(action)

        self._get_observation()
        self._get_info()

        if self.render_mode != 'none':
            self._render()

    # execute actions of all non-live agents, autoplays entire game if no live agent
    def _auto_play(self):
        while not self.is_game_over and (self.live_player == -1 or self.curr_player != self.live_player):
            action = self.players[self.curr_player].action(self.info)
            self._execute_action(action)

    # get next set of cards from board, always sorted
    def _get_next_cards(self):
        cards = []
        if self.stage == Stage.BUYING:
            cards = self.property_cards[:self.num_players]
            self.property_cards = np.delete(self.property_cards, np.arange(self.num_players))
        else:
            cards = self.money_cards[:self.num_players]
            self.money_cards = np.delete(self.money_cards, np.arange(self.num_players))
        return np.sort(cards)

    def _get_observation(self):
        # board data
        self.observation = np.concatenate(([self.stage, self.last_bid], self.board))
        
        # player data, append live player first then the rest
        i = 0
        j = self.live_player
        while i < self.num_players:
            player = self.players[j]
            player_obs = np.concatenate(([player.num, player.coins, player.money, int(player.is_bidding), player.bid], player.property))
            self.observation = np.concatenate((self.observation, player_obs))
            j += 1
            if j >= self.num_players:
                j = 0
            i += 1

        self.observation = np.array(self.observation, dtype=int)
        self.observation_space = gym.spaces.Box(low=-1, high=126, shape=self.observation.shape, dtype=int)

    def _get_info(self):
        self.info = {
            'stage': self.stage,
            'last_bid': self.last_bid,
            'num_buy_property': self.num_buy_property,
            'num_sell_round': self.num_sell_round,
            'num_players': self.num_players,
            'observation': self.observation,
            'action_mask': self.get_action_mask(player_num=self.curr_player),
            'board': self.board,
            'property': self.players[self.curr_player].property
        }
    
    # get all legal moves for a player then convert to action mask form
    def get_action_mask(self, player_num=None):
        if player_num == None:
            player_num = self.live_player

        moves = []
        player = self.players[player_num]
        if self.stage == Stage.BUYING:
            moves.append(Action.TAKE.value)
            if not self._is_last_card():
                for i in range(Action.BID_2):
                    new_bid = Action.BID_1 + i + self.last_bid
                    if new_bid <= player.coins:
                        moves.append(Action.BID_1 + i)
        else:
            for i, property in enumerate(player.property):
                if property != -1:
                    moves.append(Action.SELL_1 + i)

        action_mask = [False] * len(Action)
        for move in moves:
            action_mask[move] = True
        return action_mask

    def _resolve_illegal_action(self, action):
        if self.curr_player == self.live_player:
            print(f'illegal action {action}')
            self.reward = -1
            self.is_game_over = True

    def _is_last_card(self):
        return self.num_buy_property == self.num_players - 1

    def _resolve_buy_action(self, action):
        player = self.players[self.curr_player]

        if action == Action.TAKE:
            # pay full bid if last card else pay half rounded up
            if self._is_last_card():
                player.coins -= player.bid
            else:
                player.coins -= ceil(player.bid / 2)
            player.bid = 0
            player.is_bidding = False
            # remove lowest card from board into players properties
            card = self.board[self.num_buy_property]
            player.property[self.num_buy_round] = card
            self.num_buy_property += 1
        elif action >= Action.BID_1 and action <= Action.BID_2:
            if self.last_bid + action > player.coins or self._is_last_card():
                # illegal bid
                return self._resolve_illegal_action(action)
            player.bid = self.last_bid + action
            self.last_bid = player.bid
        else:
            # illegal buy action
            return self._resolve_illegal_action(action)

        if self.num_buy_property == self.num_players:
            # check for next stage or next round, last person to take starts next buy round
            if len(self.property_cards) == 0:
                self.stage = Stage.SELLING
                self.curr_player = self.live_player
                for player in self.players:
                    player.property.sort()
            else:
                self.num_buy_round += 1
                for player in self.players:
                    player.is_bidding = True
            self.num_buy_property = 0
            self.last_bid = 0
            self.board = self._get_next_cards()
        else:
            # cycle to next active player in round
            i = self.curr_player
            while True:
                i += 1
                if i >= self.num_players:
                    i = 0
                if self.players[i].is_bidding:
                    self.curr_player = i
                    break

    def _resolve_sell_action(self, action):
        player = self.players[self.curr_player]
        if action >= Action.SELL_1 and action <= Action.SELL_8:
            sell_index = action - Action.SELL_1
            if player.property[sell_index] == -1:
                # card already sold
                return self._resolve_illegal_action(action)
            player.sell_property = sell_index
            self.num_sell_property += 1
        else:
            # illegal sell action
            return self._resolve_illegal_action(action)

        self.sell_round_str = None
        if self.num_sell_property == self.num_players:
            # all cards have been selected
            selected_cards = []
            for player in self.players:
                selected_cards.append((player.property[player.sell_property], player.num))
                player.property[player.sell_property] = -1
                player.sell_property = -1
            
            # distribute rewards
            self.sell_round_str = 'Sell Round {}: '.format(self.num_sell_round + 1)
            selected_cards.sort(key=lambda x : x[0])
            for i, vals in enumerate(selected_cards):
                self.players[vals[1]].money += self.board[i]
                self.sell_round_str += 'P{} C:{} M:+{} | '.format(vals[1], vals[0], self.board[i])

            self.num_sell_round += 1
            self.num_sell_property = 0
            self.curr_player = self.live_player

            if len(self.money_cards) == 0:
                self.is_game_over = True
                self.board = []
                # winner has most money, tiebreaker is leftover coins
                winner = sorted(self.players, key=lambda x : (x.coins + x.money, x.coins))[-1]
                self.reward = 1 if winner.num == self.live_player else -1
            else:
                self.board = self._get_next_cards()
        else:
            # cycle to next player for card selection
            i = self.curr_player
            while True:
                i += 1
                if i >= self.num_players:
                    i = 0
                if self.players[i].sell_property == -1:
                    self.curr_player = i
                    break

    def _render(self):
        if self.render_mode == 'text':
            is_first_sell_round = self.num_sell_round == 0 and self.num_sell_property == 0
            if self.stage == Stage.BUYING or is_first_sell_round:
                if self.action != None:
                    text = 'P{} performs {} '.format(self.prev_player, str(Action(self.action)))
                    if self.action == Action.TAKE:
                        i = self.num_buy_round if self.num_buy_property != 0 else self.num_buy_round - 1
                        text += 'Card {}'.format(self.players[self.prev_player].property[i])
                    else:
                        text += 'Total Bid {}'.format(self.players[self.prev_player].bid)
                    print(text, '\n')
                print(self.board, self.players)
            else:
                if self.sell_round_str != None:
                    print(self.sell_round_str, '\n')
                    print(self.board, self.players)
                if self.is_game_over:
                    winner = sorted(self.players, key=lambda x : (x.coins + x.money, x.coins))[-1]
                    print('Player P{} wins the game! {}'.format(winner.num, [x.coins + x.money for x in self.players]))
