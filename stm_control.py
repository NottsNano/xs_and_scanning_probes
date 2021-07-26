import warnings

import numpy as np
from matplotlib import pyplot as plt
from nOmicron.mate import objects as mo
from nOmicron.microscope import IO
from nOmicron.utils.plotting import nanomap

import utils
from binarisation import ImagePreprocessing
from model.self_play_test import SelfPlayTester
from shapes.shapes import DataShape


class STMTicTacToe:
    def __init__(self, scan_bias, scan_setpoint, desorption_bias, desorption_current, t_raster, raster_points,
                 player_1_type="rules", player_2_type="best", first_player="player_1", render_mode="plot",
                 savefig=None):
        """Play noughts and crosses in STM using RL

        Parameters
        ----------
        scan_bias: float
            Scan voltage bias (Volts)
        scan_setpoint: float
            Scan setpoint current (Amps)
        desorption_bias: float
            Voltage to use during passivation (Volts)
        desorption_current: float
            Current to use during passivation (Amps)
        t_raster: float
            Raster time for use during passivation (Seconds)
        raster_points: int
            Number of sub-points to use in passivation (to replicate atom manipulation window)
        player_1_type: str
            Player 1. One of 'best', 'mostly_best', 'random', 'human' (default), 'rules', '<model>.zip'
        player_2_type: str
            Player 2. One of 'best' (default), 'mostly_best', 'random', 'human', 'rules', '<model>.zip'
        first_player: str
            Who goes first. One of 'player_1' (default), 'player_2', 'random'
        render_mode: str or None
            How to render the game. One of 'plot' (default), 'print', None
        savefig: str or None
            If we should save each figure. Either None (default), or a path to a directory
        """

        # Connect to the probe
        IO.connect()

        # STM parameters
        self.scan_bias = scan_bias
        self.scan_setpoint = scan_setpoint
        self.desorption_bias = desorption_bias
        self.desorption_current = desorption_current
        self.t_raster = t_raster
        self.raster_points = raster_points
        self.num_coarse_moves_on_reset = 5
        mo.xy_scanner.Return_To_Stored_Position(False)

        # Game parameters
        self.game = None
        self.game_args = {"player_1_type": player_1_type,
                          "player_2_type": player_2_type,
                          "first_player": first_player,
                          "render_mode": render_mode,
                          "auto_render": False}

        self.preprocessor = None

        # CNN parameters
        self.cnn_datadir = None
        # self.
        self.cnn_folder = None

        # self.ensemble_classifier = EnsembleClassifier(5, '29-12-19', ['CNN1DBatchnorm'], ['adam'], [0.001])
        # self.ensemble_classifier.load_models(rootdir="")

        # Others
        self.fig = None
        self.axs = None
        self.savefig = savefig
        self.savefig_step = 0

    def play_game(self):
        self.reset()
        while not self.game.is_episode_done:
            self.step()
        self.game._announce_winner()

    def CNN_assess(self):
        raise NotImplementedError

    def step(self):
        action = self.game.player_0.choose_action(self.game.env, choose_best_action=True, mask_invalid_actions=True)

        cross_move = DataShape("cross", centre_offset=utils.action2ind(action))
        cross_move.draw_in_stm(self.desorption_bias, self.desorption_current, self.t_raster, self.raster_points)
        cross_image = utils.get_scan()
        binarised_cross_image = self.preprocessor.preprocess_and_binarise(cross_image)
        self.render(cross_image[0, :, :], binarised_cross_image, cross_move)

        self.game.step(action)

        nought_move = DataShape("nought", centre_offset=utils.action2ind(self.game.env.player_1_last_move))
        nought_move.draw_in_stm(self.desorption_bias, self.desorption_current, self.t_raster, self.raster_points)
        nought_image = utils.get_scan()
        binarised_nought_image = self.preprocessor.preprocess_and_binarise(nought_image)
        self.render(nought_image[0, :, :], binarised_nought_image, nought_move)

    def reset(self):
        # Reset env
        self.game = SelfPlayTester(**self.game_args)
        self.preprocessor = ImagePreprocessing()

        mo.gap_voltage_control.Voltage(self.scan_bias)
        mo.regulator.Setpoint_1(self.scan_setpoint)

        # Coarse move random walk
        mo.experiment.stop()
        # print("Coarse Moving")
        # black_box.backward()
        #
        # for i in range(self.num_coarse_moves_on_reset):
        #     rand_coarse_move = np.random.rand()
        #     if 0 <= rand_coarse_move <= 0.25:
        #         black_box.x_minus()
        #     elif 0.25 < rand_coarse_move <= 0.5:
        #         black_box.x_plus()
        #     elif 0.5 < rand_coarse_move <= 0.75:
        #         black_box.y_minus()
        #     else:
        #         black_box.y_plus()
        #
        # black_box.auto_approach()

        # Check if flat
        is_flat = True  # TODO
        if not is_flat:
            warnings.warn("Game area is not viable! Retracting and moving!")
            self.reset()

        # Setup figs
        self.fig, self.axs = plt.subplots(1, 4)
        self.fig.set_size_inches(16, 6)

        self.axs[0].set_title("Game Board")
        self.axs[1].set_title("Desorption Path")
        self.axs[2].set_title("STM Image")
        self.axs[3].set_title("Binarised STM Image")
        self.axs[1].invert_xaxis()

        for i in range(4):
            self.axs[i].set_xticks([])
            self.axs[i].set_yticks([])

        self.game.env._make_axis(ax=self.axs[0])
        self.game.env.fig = self.fig
        plt.pause(0.01)

        # Draw grid
        # prelim_scan = utils.get_scan()
        # self.render(scan_data=prelim_scan[0, :, :], binary_data=prelim_scan, piece=None)
        #
        # grid = DataShape("board")
        # grid.draw_in_stm(desorb_voltage=self.desorption_bias, desorb_current=self.desorption_current,
        #                  t_raster=self.t_raster, points=self.raster_points)
        #
        # scan = utils.get_scan()
        # self.render(scan_data=scan[0, :, :], binary_data=scan, piece=grid)

    def render(self, scan_data: np.ndarray, binary_data: np.ndarray, piece: DataShape):
        self.game.env.render()
        if piece is not None:
            piece.plot(ax=self.axs[1])
        self.axs[2].imshow(np.fliplr(scan_data), cmap=nanomap)
        # self.axs[3].imshow(binary_data, cmap=utils.rabanimap)
        plt.pause(0.001)

        if self.savefig:
            plt.savefig(self.fig, f"{self.savefig}/tictactoe_{self.savefig_step}.png")
            self.savefig_step += 1


if __name__ == '__main__':
    # IO.connect()
    # test_cross = DataShape("board", centre_offset=[0, 0])
    # test_cross.draw_in_stm(desorb_voltage=4.85, desorb_current=1.2e-9,
    #                        t_raster=40e-3, points=256)


    game = STMTicTacToe(scan_bias=-2.25, scan_setpoint=250e-12,
                        desorption_bias=4.2, desorption_current=1.5e-9,
                        t_raster=20e-3, raster_points=512,
                        savefig="scans/2/")
    game.play_game()
