from time import time
import logging


_log = logging.getLogger(__name__)


class Timer:
    def __init__(self, title: str):
        self._title = title

    def add_to_title(self, text: str):
        self._title += " " + text

    def __enter__(self):
        self._t0 = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            t = time()

            dt = t - self._t0

            _log.debug(f"{dt} seconds to run {self._title}")
