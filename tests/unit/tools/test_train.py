from swiftmhc.tools.train import EarlyStopper


def test_early_stop():
    """
    Must stop early on a horizontal loss curve with outliers
    """

    eps = 0.01
    early_stop = EarlyStopper(epsilon=eps, patience=30)

    losses = [0.0] * 100

    # add outliers
    for i in range(0, len(losses), 20):
        losses[i] = 1.0

    stopped_early = False
    for loss in losses:
        early_stop.update(loss)

        if early_stop.stops_early():
            stopped_early = True
            break

    assert stopped_early


def test_no_early_stop():
    """
    Must not stop early if losses are going down
    """

    eps = 0.01
    early_stop = EarlyStopper(epsilon=eps)

    # losses going down
    losses = [0.025 * i for i in reversed(range(5, 100))]

    stopped_early = False
    for loss in losses:
        early_stop.update(loss)

        if early_stop.stops_early():
            stopped_early = True
            break

    assert not stopped_early
