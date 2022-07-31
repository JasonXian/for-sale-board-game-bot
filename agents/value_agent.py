import numpy as np

from env import Player, Action, Stage

# agent playing the game via value relative to cost
class ValueAgent(Player):
    AVG_VALS = {
        3: {
            'property': 18.5,
            'money': 9.33
        },
        4: {
            'property': 16.5,
            'money': 8.5
        },
        5: {
            'property': 15.5,
            'money': 7.93
        },
        6: {
            'property': 15.5,
            'money': 7.93
        }
    }

    def __init__(self, player_num):
        super().__init__(player_num)

    def action(self, info):
        num_players = info['num_players']
        board_val = sum(info['board']) / num_players
        if info['stage'] == Stage.BUYING:
            # if the avg property value is below, bid more since future streets
            # we can get more value for less money, but don't overpay for the property relative to how much it could earn
            avg_val = self.AVG_VALS[num_players]['property']
            last_bid = info['last_bid']
            min_profit = 7 # we need to profit this much from the card
            prop_val = info['board'][-1] / 2 - min_profit 
            prop_val += 1 if board_val <= avg_val else -1
            
            if prop_val > last_bid and info['action_mask'][1]:
                # bid if good value and its a legal move
                return Action.BID_1

            return Action.TAKE
        else:
            # find the card closest to the board in value
            card = (0, float('inf'))
            for i, prop in enumerate(info['property']):
                if prop == -1:
                    continue
                if abs(prop / 2 - board_val) < card[1]:
                    card = (i, abs(prop - board_val))

            return Action.SELL_1 + card[0]

"""
some numbers to reason about optimal plays per round
Num players: 3 | avg property value per round: 18.5 | avg money value per round 9.33 | avg dollar earned per 1 property value 0.5
Distribution of property cards: [12.27, 18.48, 24.75]
Distribution of money cards: [5.91, 9.48, 12.61]

Num players: 4 | avg property value per round: 16.5 | avg money value per round 8.5 | avg dollar earned per 1 property value 0.52
Distribution of property cards: [7.8, 13.59, 19.43, 25.18]
Distribution of money cards: [4.18, 7.05, 9.97, 12.8]

Num players: 5 | avg property value per round: 15.5 | avg money value per round 7.93 | avg dollar earned per 1 property value 0.51
Distribution of property cards: [5.21, 10.3, 15.43, 20.69, 25.87]
Distribution of money cards: [2.54, 5.37, 8.02, 10.61, 13.13]

Num players: 6 | avg property value per round: 15.5 | avg money value per round 7.93 | avg dollar earned per 1 property value 0.51
Distribution of property cards: [4.45, 8.83, 13.25, 17.75, 22.16, 26.55]
Distribution of money cards: [2.13, 4.64, 6.89, 9.11, 11.33, 13.51]
"""
def find_averages(num_players=3, num_cycles=10000):
    # run a bunch of cycles to find the avg value of cards
    NUM_CARDS = {
        3: 24,
        4: 28,
        5: 30,
        6: 30
    }
    property_cards = np.arange(1, 31)
    money_cards = np.concatenate([[0, 0], np.arange(2, 16), np.arange(2, 16)])
    while len(property_cards) != NUM_CARDS[num_players]:
        property_cards = np.delete(property_cards, 0)
        money_cards = np.delete(money_cards, 0)

    # run cycles
    dist_prop_cards = [0] * num_players
    dist_mon_cards = [0] * num_players
    for _ in range(num_cycles):
        np.random.shuffle(property_cards)
        np.random.shuffle(money_cards)
        rounds = NUM_CARDS[num_players] // num_players
        prop_cards = [0] * num_players
        mon_cards = [0] * num_players

        for i in range(rounds):
            j = i * num_players
            curr_prop_cards = sorted(property_cards[j:j+num_players])
            curr_money_cards = sorted(money_cards[j:j+num_players])
            for k in range(num_players):
                prop_cards[k] += curr_prop_cards[k]
                mon_cards[k] += curr_money_cards[k]

        for i in range(num_players):
            dist_prop_cards[i] += prop_cards[i] / rounds
            dist_mon_cards[i] += mon_cards[i] / rounds

    dist_prop_cards = [round(x / num_cycles, 2) for x in dist_prop_cards]
    dist_mon_cards = [round(x / num_cycles, 2) for x in  dist_mon_cards]
    agg_prop_val = round(sum(dist_prop_cards) / num_players, 2)
    agg_money_val = round(sum(dist_mon_cards) / num_players, 2)
    money_per_prop = round(agg_money_val / agg_prop_val, 2)

    print("Num players: {} | avg property value per round: {} | avg money value per round {} | avg dollar earned per 1 property value {} ".format(num_players, agg_prop_val, agg_money_val, money_per_prop))
    print("Distribution of property cards: {}".format(dist_prop_cards))
    print("Distribution of money cards: {} \n".format(dist_mon_cards))

if __name__ == "__main__":
    find_averages(3)
    find_averages(4)
    find_averages(5)
    find_averages(6)
