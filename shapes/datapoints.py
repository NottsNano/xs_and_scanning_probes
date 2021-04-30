import numpy as np
from typing import Tuple
import json
from matplotlib import pyplot as plt
import ast

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
