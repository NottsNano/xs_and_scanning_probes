from shapes import DataShape


class Nought(DataShape):
    def __init__(self):
        super().__init__("nought")


class Cross(DataShape):
    def __init__(self):
        super().__init__("cross")


class GameBoard(DataShape):
    def __init__(self):
        super().__init__("board")
