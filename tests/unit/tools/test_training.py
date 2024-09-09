from swiftmhc.tools.train import EarlyStopper


def test_early_stop():

    eps = 0.01
    early_stop = EarlyStopper(epsilon=eps)

    losses = [0.0] * 100

    stopped_early = False
    for loss in losses:
        early_stop.update(loss)

        if early_stop.stops_early():
            stopped_early = True
            break

    assert stopped_early


def test_no_early_stop():

    eps = 0.01
    early_stop = EarlyStopper(epsilon=eps)

    losses = [0.0, eps + 0.001] * 50

    stopped_early = False
    for loss in losses:
        early_stop.update(loss)

        if early_stop.stops_early():
            stopped_early = True
            break

    assert not stopped_early
