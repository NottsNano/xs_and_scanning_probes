import warnings

import numpy as np
from matplotlib import pyplot as plt
from nOmicron.utils.plotting import nanomap

import utils
from binarisation import ImagePreprocessing
from model.self_play_test import SelfPlayTester
from shapes.shapes import DataShape

# from nOmicron.microscope import IO, xy_scanner, black_box



class STMTicTacToe:
    def __init__(self, scan_bias, desorption_bias,
                 player_1_type="human", player_2_type="best", first_player="player_1", render_mode="plot"):
        """Play noughts and crosses in STM using RL

        Parameters
        ----------
        scan_bias: float
            Scan voltage bias (Volts)
        desorption_bias: float
            Voltage to use during passivation (Volts)
        player_1_type: str
            Player 1. One of 'best', 'mostly_best', 'random', 'human' (default), 'rules', '<model>.zip'
        player_2_type: str
            Player 2. One of 'best' (default), 'mostly_best', 'random', 'human', 'rules', '<model>.zip'
        first_player: str
            Who goes first. One of 'player_1' (default), 'player_2', 'random'
        render_mode: str or None
            How to render the game. One of 'plot' (default), 'print', None
        """

        # Connect to the probe
        # IO.connect()

        # STM parameters
        self.scan_bias = scan_bias
        self.desorption_bias = desorption_bias
        self.num_coarse_moves_on_reset = 2

        # Game parameters
        self.game = None
        self.game_args = {"player_1_type": player_1_type,
                          "player_2_type": player_2_type,
                          "first_player": first_player,
                          "render_mode": render_mode,
                          "auto_render": False}

        self.preprocessor = None

        # Others
        self.fig = None
        self.axs = None

        self.reset()

    def play_game(self):
        while not self.game.is_episode_done:
            self.step()
        self.game._announce_winner()

    def step(self):
        action = self.game.player_0.choose_action(self.game.env, choose_best_action=True, mask_invalid_actions=True)

        cross_move = DataShape("cross", centre_offset=utils.action2ind(action))
        cross_move.draw_in_stm(self.desorption_bias)
        cross_image = utils.get_scan()
        binarised_cross_image = self.preprocessor.preprocess_and_binarise(cross_image)
        self.render(cross_image, binarised_cross_image, cross_move)

        self.game.step(action)

        nought_move = DataShape("nought", centre_offset=utils.action2ind(self.game.env.player_1_last_move))
        nought_move.draw_in_stm(self.desorption_bias)
        nought_image = utils.get_scan()
        binarised_nought_image = self.preprocessor.preprocess_and_binarise(nought_image)
        self.render(nought_image, binarised_nought_image, nought_move)

    def reset(self):
        # Reset env
        self.game = SelfPlayTester(**self.game_args)
        self.preprocessor = ImagePreprocessing()

        # Coarse move random walk
        # black_box.backward()

        # for i in range(self.num_coarse_moves_on_reset):
        # rand_coarse_move = np.random.rand()
        # if 0 <= rand_coarse_move <= 0.25:
        #     black_box.x_minus()
        # elif 0.25 < rand_coarse_move <= 0.5:
        #     black_box.x_plus()
        # elif 0.5 < rand_coarse_move <= 0.75:
        #     black_box.y_minus()
        # else:
        #     black_box.y_plus()

        # black_box.auto_approach()

        # Take a single scan
        # prelim_scan = self.get_scan()
        # plt.imshow(prelim_scan, cmap=nanomap, axs=self.axs[2])

        # Check if flat
        is_flat = True  # TODO
        if not is_flat:
            warnings.warn("Game area is not viable! Retracting and moving!")
            self.reset()

        # Draw grid
        grid = DataShape("board")
        grid.draw_in_stm(desorb_voltage=self.desorption_bias)

        # Setup figs
        self.fig, self.axs = plt.subplots(1, 4)

        self.axs[0].set_title("Agent game")
        self.axs[1].set_title("Desorption path")
        self.axs[2].set_title("STM image")
        self.axs[3].set_title("Binarised STM image")

        self.axs[1].invert_yaxis()
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])

        self.game.env._make_axis(ax=self.axs[0])
        self.game.env.fig = self.fig

        scan = utils.get_scan()
        binarised_scan = self.preprocessor.preprocess_and_binarise(scan)
        self.render(scan_data=scan, binary_data=binarised_scan, piece=grid)

    def render(self, scan_data: np.ndarray, binary_data: np.ndarray, piece: DataShape):
        self.game.env.render()
        piece.plot(ax=self.axs[1])
        self.axs[2].imshow(scan_data, cmap=nanomap)
        self.axs[3].imshow(binary_data, cmap=utils.rabanimap)
        plt.pause(0.001)


if __name__ == '__main__':
    # Parse args
    game = STMTicTacToe(scan_bias=0, desorption_bias=0)
    game.play_game()
