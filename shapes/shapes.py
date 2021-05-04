import numpy as np
from typing import Tuple
import json
from matplotlib import pyplot as plt
import ast


# Routine:
# Locate centres of
# Desorb hydorgen

# Draw grid

# Function to draw line at vertices (relative to 512x512 grid for simplicity)


class DataPoint(object):
    def __init__(self, pos: Tuple[int, int], desorb_on_approach: bool):
        self.pos = pos
        self.desorb_on_approach = desorb_on_approach


class DataShape(object):
    def __init__(self, object_shape: str, shape_directory="data/"):
        self.shape_directory = shape_directory
        self.object_shape = object_shape

        self.datafile = None

        self.centre_offset = None
        self.size = None
        self.datapoints = []

        self.load_file()
        self.prepare()

    def load_file(self):
        with open(f"{self.shape_directory}{self.object_shape}.json") as f:
            self.datafile = json.load(f)

    def prepare(self):
        self.centre_offset = self.datafile["centre_offset"]
        self.size = self.datafile["size"]
        for point in self.datafile["all_points"]:
            if np.any(np.array(point["datapoint"]) < 0) or np.all(np.array(point["datapoint"]) > self.size):
                raise LookupError("Point is outside bounds of shape")
            self._add_datapoint(point["datapoint"], ast.literal_eval(point["desorb"]))

    def _add_datapoint(self, xy, desorb_on_approach):
        self.datapoints.append(DataPoint(xy, desorb_on_approach))

    def plot(self, ax=None):
        if not ax:
            fig, ax = plt.subplots(1, 1)

            padding = self.size // 50
            ax.set_xlim(-padding, self.size + padding)
            ax.set_ylim(-padding, self.size + padding)
            ax.set_title(self.object_shape)
            ax.set_aspect(1)

        for i in range(len(self.datapoints) - 1):
            xs = [self.datapoints[i].pos[0], self.datapoints[i + 1].pos[0]]
            ys = [self.datapoints[i].pos[1], self.datapoints[i + 1].pos[1]]
            if self.datapoints[i + 1].desorb_on_approach:
                ax.plot(xs, ys, 'g')
            else:
                ax.plot(xs, ys, 'r')

    def int2mtrx(self):
        pass  # convert from 0-512 to -1 - 1

    def draw(self, pos):
        """Draws the object, centered at pos"""
        pass
    # assert pos is int not mtrx coord!


