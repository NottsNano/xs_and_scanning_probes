import os
from time import sleep

import numpy as np
import tensorflow as tf

from SIMPLE.utils.agents import Agent
from SIMPLE.utils.files import load_model
from SIMPLE.utils.register import get_environment
from SIMPLE.utils.selfplay import selfplay_wrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SelfPlayTester:
    def __init__(self, player_1_type="human", player_2_type="best", first_player="player_1", render_mode="plot",
                 rand_seed=1234):
        """Plays the tic tac toe env

        Parameters
        ----------
        player_1_type: str
            Player 1. One of 'best', 'mostly_best', 'random', 'human' (default), 'rules', '<model>.zip'
        player_2_type: str
            Player 2. One of 'best' (default), 'mostly_best', 'random', 'human', 'rules', '<model>.zip'
        first_player: str
            Who goes first. One of 'player_1' (default), 'player_2', 'random'
        render_mode: str or None
            How to render the game. One of 'plot' (default), 'print', None
        rand_seed: int
            Seed for the random number generator. Default 1234
        """

        self.env = None
        self.player_1_type = player_1_type
        self.player_2_type = player_2_type
        self.first_player = first_player
        self.render_mode = render_mode
        self.rand_seed = rand_seed
        self.player_0 = None
        self.is_episode_done = False

        self._setup_env()

    def _setup_env(self):
        base_env = get_environment("tictactoe")
        self.env = selfplay_wrapper(base_env)(opponent_type=self.player_2_type, first_player=self.first_player,
                                              render_mode=self.render_mode,
                                              verbose=False, deploy_mode="test")
        self.env.seed(self.rand_seed)

        if "zip" in self.player_1_type:
            self.player_0 = Agent('PPO_Agent', load_model(self.env, self.player_1_type))
        else:
            self.player_0 = Agent(self.player_1_type)

        self.reset()

    def play_game(self):
        self.reset()
        while not self.is_episode_done:
            self.step()

    def announce_winner(self):
        if self.env.check_game_over()[0] == 0 and self.env.check_game_over()[0] == True:
            print("Draw!")
        else:
            print(f"Winner = {self.env.current_player.token.symbol}")

    def reset(self):
        self.env.reset()

    def step(self):
        self.env.render()
        action = self.player_0.choose_action(self.env, choose_best_action=True, mask_invalid_actions=True)

        obs, reward, self.is_episode_done, _ = self.env.unwrapped.step(action)

        if self.is_episode_done:
            self.env.render()


if __name__ == '__main__':
    game = SelfPlayTester()
    game.play_game()
