

import numpy


class EarlyStopper:
    """
    Keeps track of early stopping variables and checks whether the conditions
    for early stopping are met.
    """

    def __init__(self, patience: int = 15, epsilon: float = 0.01):

        # number of successive epochs matching epsilon
        self._patience = patience

        # max difference between validation loss values
        self._epsilon = epsilon

        # remembers the previous loss values
        self._record = []

    def update(self, validation_loss: float):

        self._record.append(validation_loss)

    def stops_early(self) -> bool:

        if len(self._record) > self._patience:

            recent = numpy.array(self._record[-self._patience:])

            return numpy.all(numpy.abs(recent - recent.mean()) < self._epsilon)
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
        fape_tune: bool,
        torsion_tune: bool,
        affinity_tune: bool,
        fine_tune: bool,
    ):
        self._epoch_count = 0
        self._max_epoch_count = max_epoch_count
        self._fape_tune = fape_tune
        self._torsion_tune = torsion_tune
        self._affinity_tune = affinity_tune
        self._fine_tune = fine_tune

    def end_reached(self) -> bool:
        return self._epoch_count >= self._max_epoch_count

    def update(self):
        self._epoch_count += 1

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
    def max_epoch_count(self):
        return self._max_epoch_count

    def __str__(self):
        return f"max_epoch={self._max_epoch_count}, fape_tune={self._fape_tune}, torsion_tune={self._torsion_tune}, affinity_tune={self._affinity_tune}, fine_tune={self._fine_tune}"

