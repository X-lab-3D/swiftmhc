from typing import List

import numpy


class EarlyStopper:
    """
    Keeps track of early stopping variables and checks whether the conditions
    for early stopping are met.
    """

    def __init__(self, patience: int = 50, epsilon: float = 1e-4):

        # number of successive epochs matching epsilon
        self._patience = patience

        # max difference between validation loss values
        self._epsilon = epsilon

        # remembers the previous loss values
        self._record = []

    @staticmethod
    def _filter_outliers(values: List[float]) -> List[float]:

        if len(values) == 0:
            return values

        if all([value == values[0] for value in values]):
            # all the same
            return values

        # define min & max
        median = numpy.median(values)
        q1 = numpy.quantile(values, 0.25)
        q3 = numpy.quantile(values, 0.75)

        iqr = q3 - q1

        min_ = q1 - 1.5 * iqr
        max_ = q3 + 1.5 * iqr

        # filter
        passed = []
        for value in values:
            if value >= min_ and value <= max_:
                passed.append(value)
            else:
                passed.append(median)

        return passed


    def update(self, validation_loss: float):

        self._record.append(validation_loss)

    def stops_early(self) -> bool:

        if len(self._record) > self._patience:

            recent = numpy.array(self._filter_outliers(self._record[-self._patience:]))
            if recent.shape[0] < 2:
                return False

            # all the same?
            if numpy.all(recent == recent[0].item()):
                return True

            epochs = numpy.arange(0, recent.shape[0])

            slope, offset = numpy.polyfit(epochs, recent, 1)

            return numpy.abs(slope) < self._epsilon
        else:
            return False


class TrainingPhase:
    """
    Keeps track of the number of epochs passed,
    before a maximum is reached.
    """

    def __init__(
        self,
        max_epoch_count: int,
        lr: float,
        fape_tune: bool,
        torsion_tune: bool,
        affinity_tune: bool,
        fine_tune: bool,
    ):
        self._epoch_count = 0
        self._max_epoch_count = max_epoch_count
        self._lr = lr
        self._fape_tune = fape_tune
        self._torsion_tune = torsion_tune
        self._affinity_tune = affinity_tune
        self._fine_tune = fine_tune

    def end_reached(self) -> bool:
        return self._epoch_count >= self._max_epoch_count

    def update(self):
        self._epoch_count += 1

    @property
    def max_epoch_count(self) -> int:
        return self._max_epoch_count

    @property
    def epoch_count(self) -> int:
        return self._epoch_count

    @property
    def fape_tune(self):
        return self._fape_tune

    @property
    def torsion_tune(self):
        return self._torsion_tune

    @property
    def affinity_tune(self):
        return self._affinity_tune

    @property
    def fine_tune(self):
        return self._fine_tune

    @property
    def lr(self):
        return self._lr

        return self._max_epoch_count

    def __str__(self):
        return f"max_epoch={self._max_epoch_count}, lr={self._lr}, fape_tune={self._fape_tune}, torsion_tune={self._torsion_tune}, affinity_tune={self._affinity_tune}, fine_tune={self._fine_tune}"

