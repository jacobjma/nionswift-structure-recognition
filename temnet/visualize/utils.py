import asyncio
from time import time

import matplotlib
import numpy as np


class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()


def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """

    def decorator(fn):
        timer = None

        def debounced(*args, **kwargs):
            nonlocal timer

            def call_it():
                fn(*args, **kwargs)

            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)

        return debounced

    return decorator


def throttle(wait):
    """ Decorator that prevents a function from being called
        more than once every wait period. """

    def decorator(fn):
        time_of_last_call = 0
        scheduled = False
        new_args, new_kwargs = None, None

        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled

            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False

            time_since_last_call = time() - time_of_last_call
            new_args = args
            new_kwargs = kwargs
            if not scheduled:
                new_wait = max(0, wait - time_since_last_call)
                Timer(new_wait, call_it)
                scheduled = True

        return throttled

    return decorator


def get_colors_from_cmap(c, cmap=None, vmin=None, vmax=None):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('viridis')

    elif isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    if vmin is None:
        vmin = np.nanmin(c)

    if vmax is None:
        vmax = np.nanmax(c)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    c = np.array(c, dtype=float)

    valid = np.isnan(c) == 0
    colors = np.zeros((len(c), 4))
    colors[valid] = cmap(norm(c[valid]))

    return colors