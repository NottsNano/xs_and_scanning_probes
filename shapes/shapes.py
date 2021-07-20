import warnings

import numpy as np
from typing import Tuple
import json
from matplotlib import pyplot as plt
import ast
from nOmicron.mate import objects as mo

from utils import ind2mtrx


class DataPoint(object):
    def __init__(self, pos: Tuple[int, int], desorb_on_approach: bool):
        self.pos = pos
        self.desorb_on_approach = desorb_on_approach

    def move_to_point(self, desorb_voltage, desorb_current, t_raster, points):
        # Store old parameters
        old_voltage = mo.gap_voltage_control.Voltage()
        old_current = mo.regulator.Setpoint_1()
        old_raster = mo.xy_scanner.Raster_Time()

        # Prep for movement
        new_raster = t_raster * points / mo.xy_scanner.Points()
        mo.xy_scanner.Raster_Time(new_raster)
        if self.desorb_on_approach:
            mo.gap_voltage_control.Voltage(desorb_voltage)
            mo.regulator.Setpoint_1(desorb_current)

        # Do movement
        mo.xy_scanner.Target_Position(ind2mtrx(self.pos))
        mo.xy_scanner.move()

        # Reset
        mo.gap_voltage_control.Voltage(old_voltage)
        mo.regulator.Setpoint_1(old_current)
        mo.xy_scanner.Raster_Time(old_raster)


class DataShape(object):
    def __init__(self, object_shape: str, centre_offset=[0, 0], shape_directory="shapes/data/"):
        self.shape_directory = shape_directory
        self.object_shape = object_shape

        self.datafile = None

        self.centre_offset = np.array(centre_offset)
        self.size = None
        self.datapoints = []

        self.load_file()
        self.prepare()

    def load_file(self):
        with open(f"{self.shape_directory}{self.object_shape}.json") as f:
            self.datafile = json.load(f)

    def prepare(self):
        self.centre_offset += self.datafile["centre_offset"]
        self.size = self.datafile["size"]
        for point in self.datafile["all_points"]:
            if np.any(np.array(point["datapoint"]) < 0) or np.all(np.array(point["datapoint"]) > self.size):
                raise LookupError("Point is outside bounds of shape")
            self._add_datapoint(point["datapoint"], ast.literal_eval(point["desorb"]))

    def _add_datapoint(self, xy: np.ndarray, desorb_on_approach: bool):
        self.datapoints.append(DataPoint(xy + self.centre_offset, desorb_on_approach))

    def _make_axs(self, ax):
        if not ax:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(self.object_shape)
        padding = self.size // 50
        ax.set_xlim(-padding, self.size + padding)
        ax.set_ylim(-padding, self.size + padding)
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def draw_in_stm(self, desorb_voltage, desorb_current, t_raster, points):
        mo.experiment.pause()
        for datapoint in self.datapoints:
            datapoint.move_to_point(desorb_voltage, desorb_current, t_raster, points)
        mo.experiment.resume()

    def plot(self, ax=None):
        if ax is None:
            ax = self._make_axs(ax)

        for i in range(len(self.datapoints) - 1):
            xs = np.array([self.datapoints[i].pos[0], self.datapoints[i + 1].pos[0]]) + self.centre_offset[0]
            ys = np.array([self.datapoints[i].pos[1], self.datapoints[i + 1].pos[1]]) + self.centre_offset[1]
            if self.datapoints[i + 1].desorb_on_approach:
                ax.plot(xs, ys, 'g')
            else:
                ax.plot(xs, ys, 'r')

        ax.set_aspect('equal', adjustable='box')
        # padding = self.size // 50
        # ax.set_xlim(-padding, self.size + padding)
        # ax.set_ylim(-padding, self.size + padding)
        # ax.set_xticks([])
        # ax.set_yticks([])
