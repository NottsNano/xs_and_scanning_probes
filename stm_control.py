from nOmicron.microscope import IO, xy_scanner

class STMControl():
    def __init__(self):
        # Connect to the microscope
        IO.connect()

        # Set up the game


    def get_scan(self):
        img = xy_scanner.get_xy_scan("Z", "Forward", "Up")

if __name__ == '__main__':
    game = STMControl()