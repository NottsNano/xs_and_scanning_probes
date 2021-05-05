import warnings

import numpy as np
from skimage import filters


class ImagePreprocessing:
    def __init__(self):
        self.are_flattening_parameters_set = False
        self.prelim_image = None
        self.poly_flat_order = None
        self.xv = None
        self.yv = None
        self.threshold_method = "multiotsu"

    def preprocess_and_binarise(self, image):

        return image

        norm_data = self._normalize_data(image)
        median_data = self._median_align(norm_data)

        if self.are_flattening_parameters_set:
            self.set_flattening_parameters(median_data)

        flattened_data = self.flatten(median_data)
        flattened_data = self._normalize_data(flattened_data)
        binarized_data = self._binarise(flattened_data)

        return binarized_data

    @staticmethod
    def _normalize_data(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    @staticmethod
    def _median_align(arr):
        for i in range(len(arr)):
            diff = arr[i - 1, :] - arr[i, :]
            bins = np.linspace(np.min(diff), np.max(diff), 1000)
            binned_indices = np.digitize(diff, bins, right=True)
            np.sort(binned_indices)
            median_index = np.median(binned_indices)
            arr[i, :] += bins[int(median_index)]

        return arr

    def set_flattening_parameters(self, prelim_image, n=5):
        """We don't want to desorption sites to introduce artefacts, so just do this once initially"""

        if self.are_flattening_parameters_set:
            warnings.warn("Flattening parameters have already been set for this game! Overwriting...")
        self.prelim_image = prelim_image
        self.poly_flat_order = n

        horz_mean = np.mean(self.prelim_image, axis=0)  # averages all the columns into a x direction array
        vert_mean = np.mean(self.prelim_image, axis=1)  # averages all the rows into a y direction array

        line_array = np.arange(len(self.prelim_image))

        horz_fit = np.polyfit(line_array, horz_mean, n)
        vert_fit = np.polyfit(line_array, vert_mean, n)

        horz_polyval = -np.poly1d(horz_fit)
        vert_polyval = -np.poly1d(vert_fit)

        self.xv, self.yv = np.meshgrid(horz_polyval(line_array), vert_polyval(line_array))
        self.are_flattening_parameters_set = True

    def flatten(self, image):
        if not self.are_flattening_parameters_set:
            raise RuntimeError("Flattening parameters must first be set with set_flattening_parameters")

        return image + self.yv + self.xv

    def _binarise(self, arr):
        threshes = filters.threshold_multiotsu(arr, classes=2)
        # if type(threshes) is np.ndarray:  # Some thresholds return multiple levels - reduce to 2
        #     threshes = threshes[0]
        binarised = arr > threshes

        return binarised


if __name__ == '__main__':
    pass
