from env import Player, Action, Stage

# human agent playing the game via keyboard inputs
class HumanAgent(Player):
    def __init__(self, player_num):
        super().__init__(player_num)

    def action(self, info):
        action_mask = info['action_mask']
        if info['stage'] == Stage.BUYING:
            user_str = 'Buy Actions: Exit (e) | '
            for i in range(Action.BID_2 + 1):
                if action_mask[i]:
                    user_str += f'{str(Action(i))} ({i}) | '
            return self._handle_input(user_str, int(Action.TAKE), int(Action.BID_2))
        else:
            user_str = 'Sell Actions: Exit (e) | '
            for i in range(Action.SELL_1, Action.SELL_8 + 1):
                if action_mask[i]:
                    user_str += f'{str(Action(i))} ({i}) | '
            return self._handle_input(user_str, int(Action.SELL_1), int(Action.SELL_8))

    def _handle_input(self, user_str, lower, upper):
        action = -1
        while action < lower or action > upper:
            action = input(user_str)
            if action == 'e':
                raise Exception('human agent exit')
            try:
                action = int(action)
            except:
                action = -1
                print('invalid input')
        return action
