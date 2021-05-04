from nOmicron.microscope import IO, xy_scanner

from model.self_play_test import SelfPlayTester


class STMControl():
    def __init__(self):
        # Connect to the microscope
        IO.connect()

        # Set up the game
        self.game = SelfPlayTester()

    def get_scan(self):
        img = xy_scanner.get_xy_scan("Z", "Forward", "Up")

if __name__ == '__main__':
    game = STMControl()