import warnings

# from nOmicron.microscope import IO, xy_scanner, black_box

from nOmicron.utils.plotting import nanomap

from model.self_play_test import SelfPlayTester
from shapes.shapes import DataShape
from matplotlib import pyplot as plt
import numpy as np

from utils import action2ind


class STMTicTacToe:
    def __init__(self, scan_bias, scan_setpoint, passivate_setpoint,
                 player_1_type="human", player_2_type="best", first_player="player_1", render_mode="plot"):
        """Play noughts and crosses in STM using RL

        Parameters
        ----------
        scan_bias: float
            Scan voltage bias (Volts)
        scan_setpoint: float
            Scan sepoint current (Amps)
        passivate_setpoint: float
            Setpoint current to use during passivation (Amps)
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
        self.scan_setpoint = scan_setpoint
        self.passivate_setpoint = passivate_setpoint
        self.num_coarse_moves_on_reset = 2

        # Game parameters
        self.game = None
        self.game_args = {"player_1_type": player_1_type,
                          "player_2_type": player_2_type,
                          "first_player": first_player,
                          "render_mode": render_mode,
                          "auto_render": False}

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

        self.game.step(action)

        cross_move = DataShape("cross", centre_offset=action2ind(self.game.env.player_0_last_move))
        cross_move.draw_in_stm(self.passivate_setpoint)
        cross_image = self.get_scan()
        self.render(cross_image, cross_move)

        nought_move = DataShape("nought", centre_offset=action2ind(self.game.env.player_1_last_move))
        nought_move.draw_in_stm(self.passivate_setpoint)
        nought_image = self.get_scan()
        self.render(nought_image, nought_move)

    def reset(self):
        # Reset env
        self.game = SelfPlayTester(**self.game_args)

        # Coarse move & approach
        # black_box.backward()
        #
        # rand_coarse_move = np.random.rand()
        # if 0 <= rand_coarse_move <= 0.25:
        #     move_dir = black_box.x_minus
        # elif 0.25 < rand_coarse_move <= 0.5:
        #     move_dir = black_box.x_plus
        # elif 0.5 < rand_coarse_move <= 0.75:
        #     move_dir = black_box.y_minus
        # else:
        #     move_dir = black_box.y_plus
        # for i in range(self.num_coarse_moves_on_reset):
        #     move_dir()

        # black_box.auto_approach()

        # Take a single scan
        # prelim_scan = self.get_scan()
        # plt.imshow(prelim_scan, cmap=nanomap, axs=self.axs[2])

        # Check if flat
        is_flat = True
        if not is_flat:
            warnings.warn("Game area is not viable! Retracting and moving!")
            self.reset()

        # Draw grid
        grid = DataShape("board")
        grid.draw_in_stm(desorb_voltage=self.passivate_setpoint)

        # Setup figs
        self.fig, self.axs = plt.subplots(1, 3)
        self.axs[1].invert_yaxis()
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])

        self.game.env._make_axis(ax=self.axs[0])
        self.game.env.fig = self.fig

        self.render(scan_data=self.get_scan(), piece=grid)

    @staticmethod
    def get_scan():
        return np.random.rand(512, 512)
        # return xy_scanner.get_xy_scan("Z", "Forward", "Up")

    def find_human_move(self):
        # Locate centres of grids
        return int(input())

    def render(self, scan_data: np.ndarray, piece: DataShape):
        self.game.env.render()
        piece.plot(ax=self.axs[1])
        self.axs[2].imshow(scan_data, cmap=nanomap)
        plt.pause(0.001)


if __name__ == '__main__':
    # Parse args
    game = STMTicTacToe(0,0,0)
    game.play_game()
