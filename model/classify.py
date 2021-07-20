import numpy as np
import tensorflow as tf
from tensorflow import Session, Graph
from tensorflow.keras.models import load_model


class EnsembleClassifier:
    """ Classifies any given images in the shape using the models described within.

    Parameters
    ----------
    no_cats : int
        Number of categories to be classified.
    date : string
        Date that models were trained on, in the format 'DD-MM-YY'
        self.mod_names : list of strings
        Model names to be loaded and classified.
    folder_num : int, optional
        Number of subfolder to open. Default is 1.
    mod_names : Iterable[str]
        Model names to be loaded and classified.
        These are defined in model_picker.py.
    mod_opts : Iterable[str]
        Model optimisers to be loaded and classified. One of {'adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd'}
    mod_lrates = Iterable[float]
        Model learning rates to be loaded and classified.

    Examples
    --------
    >>> import numpy as np
    >>> classifier = EnsembleClassifier(no_cats=5, date='29-01-19')
    >>> classifier.mod_names = ['CNN1DBatchnorm']
    >>> classifier.mod_opts = ['adam']
    >>> classifier.mod_lrates = [1]
    >>> classifier.load_models()
    >>> testing_set = ImportedData()
    >>> testing_set.import_data_line(f'Training_Data/UtrechtAuHoldout.mat')
    >>> preds = classifier.cnn_classify(testing_set.y_train)
    >>> ensemble_preds = classifier.ensemble_predict(preds)
    """

    def __init__(self, no_cats, date, folder_num=1, mod_names=None, mod_opts=None, mod_lrates=None, verbose=True):
        self.mod_names = mod_names
        self.mod_opts = mod_opts
        self.mod_lrates = mod_lrates
        self._date = date
        self._no_cats = no_cats
        self._model = {}
        self.folder_num = folder_num
        self.verbose = verbose

        # Make an assigned graph to allow for Coach/Stable Baselines interface
        self._graph = Graph()
        with self._graph.as_default():
            self._session = Session()

    def load_models(self, rootdir=""):
        """ Loads models into the EnsembleClassifier class.

        Returns
        -------
        None

        Examples
        --------
        >>> classifier = EnsembleClassifier(5, '29-12-19', ['CNN1DBatchnorm'], ['adam'], [0.001])
        >>> classifier.load_models()
        """

        # Load models and store in memory
        if not self.verbose:
            tf.autograph.set_verbosity(3)

        # Do this in another graph to prevent conflict with stable-baselines, coach, etc
        with self._graph.as_default():
            with self._session.as_default():
                print('    Loading Models:') if self.verbose else None
                for load_mod in range(len(self.mod_names)):
                    print(f'        Model {load_mod + 1}: {self.mod_names[load_mod]}') if self.verbose else None
                    self._model[load_mod] = load_model(
                        f'{rootdir}Data/CNNData/{self._date}/{self.folder_num}/models_{self.mod_names[load_mod]}/{self.mod_names[load_mod]}_{self.mod_opts[load_mod]}_lrate_{self.mod_lrates[load_mod]}_0_checkpoint.h5')  #

    def classify(self, test_set):
        """ Classifies all models described in the EnsembleClassifier class.

        Parameters
        ----------
        test_set : ndarray
            2D images in the shape (no_images, img_size, img_size, no_features)
            OR 1D line in the shape (no_lines, line_length, no_features)

        Returns
        -------
        total_preds : ndarray
            3D array with the shape (input data, classification category, model number).

        Examples
        --------
        >>> classifier = EnsembleClassifier(5, '29-12-19', ['CNN1DBatchnorm'], ['adam'], [0.001])
        >>> classifier.load_models()
        >>> testing_set = ImportedData()
        >>> testing_set.import_data_line(f'Training_Data/UtrechtAuHoldout.mat')
        >>> total_preds = classifier.cnn_classify(testing_set.x_test)
        """

        # Preallocate
        total_preds = np.zeros((np.size(test_set, 0), self._no_cats, len(self.mod_names)))

        # Make predictions for each model
        print(f'    Making Predictions:') if self.verbose else None
        with self._graph.as_default():
            with self._session.as_default():
                for load_mod in range(len(self.mod_names)):
                    print(
                        f'        Predicting Model {load_mod + 1}: {self.mod_names[load_mod]}') if self.verbose else None
                    # Make predictions
                    total_preds[:, :, load_mod] = (
                            total_preds[:, :, load_mod] + self._model[load_mod].predict(test_set))

        return total_preds

    def ensemble_predict(self, preds):
        """ Performs ensemble prediction.

        Parameters
        ----------
        preds : ndarray
            3D array with the shape (input data, classification category, model number).

        Returns
        -------
        ensemble_preds : ndarray
            2D array with the shape (input data, classification category).
        """

        ensemble_preds = np.mean(preds, 2)
        return ensemble_preds
