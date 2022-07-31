from env import Player, Action, Stage

# agent performs the worst possible action of always taking in buy round then selling in order
class SubOptimalAgent(Player):
    def __init__(self, player_num):
        super().__init__(player_num)

    def action(self, info):
        if info['stage'] == Stage.BUYING:
            return Action.TAKE
        else:
            return Action.SELL_1 + info['num_sell_round']
            