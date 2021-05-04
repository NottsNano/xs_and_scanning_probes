from functools import partial

import gym
import numpy as np


class NoughtsAndCrossesEnv(gym.Env):
    metadata = {'render.modes': ['print', 'gui']}
    symbols = ['X', 'O', '-']
    opponents = ["Human", "AI"]

    def __init__(self, opponent, render_mode="print"):
        self.board = None
        self.action_space = gym.spaces.Discrete(9)  # 9 squares in board
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[9], dtype=np.int)
        self.current_player_num = None

        if opponent in self.opponents:
            self.opponent = opponent
        else:
            raise ValueError(f"opponent must be one of {self.opponents}")

        if render_mode in self.metadata["render.modes"]:
            self.render_mode = render_mode
        else:
            raise ValueError(f"render_mode must be one of {self.metadata['render.modes']}")

        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player_num = 0#np.random.choice([-1, 1])  # -1 = "X", 1="O"

        return self.board

    def swap_player(self):
        self.current_player_num = -self.current_player_num

    def step(self, action):
        if self.is_move_legitimate(position=action):
            self.board[action] = self.current_player_num
            if self.is_game_over():
                reward = 1
                done = True
                winner = self.symbols[self.current_player_num] + "Wins"
            else:
                reward = 0
                done = False
                winner = None
        else:  # Illegitimate action - punish and terminate
            reward = -10
            done = True
            winner = "Illegal Move"

        self.swap_player()

        return self.board, reward, done, {"end_reason": winner}

    def is_move_legitimate(self, position):
        return not self.board[position]

    def _game_over_func(self, x, a):
        return len(a[a == x]) / len(a)

    def is_game_over(self):
        board_reshaped = self.board.reshape((3, 3))
        player_1 = partial(self._game_over_func, -1)
        player_2 = partial(self._game_over_func, 1)

        data_list = [board_reshaped, board_reshaped.T, board_reshaped.diagonal(), np.flipud(board_reshaped).diagonal()]
        return any(np.apply_along_axis(f, 0, d).any() for d in data_list for f in [player_1, player_2])

    def render(self, mode='print'):
        if self.render_mode == "print":
            board = self.board.astype(str)

            board[self.board == -1] = self.symbols[0]
            board[self.board == 0] = self.symbols[2]
            board[self.board == 1] = self.symbols[1]

            print(board.reshape((3, 3)))
        else:
            raise NotImplementedError

