import functools
import logging
import time
from typing import Callable, Dict, Iterable, Optional, Set, Union

STATS_KEY_BUG: str = "bug_occurence"
DEFAULT_STATS: Set[str] = {STATS_KEY_BUG}


logger: logging.Logger = logging.getLogger(__name__)


class Stats:
    def __init__(self, counters: Iterable[str]) -> None:
        self.counters: Set[str] = set(counters)
        self.stats: Dict = {}
        self.drop_counters()

    @staticmethod
    def metered(query_type: str, obj_type: Optional[str] = None) -> Callable:
        """
        metered is a decorator used to update statistics for Patchwork's api calls.
        The stat key is infered from the decorator first argument and the method first argument when called with only 1 parameter.
        Both decorator arguments when called with 2 arguments.

        WARNING: While it is a standard decorator, it should only be applied to Patchwork method as
        it makes use of `self.set_counter` and `self.increment_counter` functions.
        """

        def metered_decorator(func):
            @functools.wraps(func)
            def metered_wrapper(*args, **kwargs):
                stats = args[0]
                obj = obj_type or args[1]

                start = time.time()
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start

                stats.set_counter(f"{obj}_{query_type}_time", elapsed_time)
                stats.increment_counter(f"{obj}_{query_type}_count")
                return result

            return metered_wrapper

        return metered_decorator

    def drop_counters(self) -> None:
        for counter in DEFAULT_STATS | self.counters:
            self.stats[counter] = 0

    def increment_counter(self, key: str, increment: int = 1) -> None:
        try:
            self.stats[key] += increment
        except Exception:
            self.stats[STATS_KEY_BUG] += 1
            logger.error(f"Failed to add {increment} increment to '{key}' stat")

    def set_counter(self, key: str, value: Union[int, float]) -> None:
        try:
            self.stats[key] = value
        except Exception:
            self.stats[STATS_KEY_BUG] += 1
            logger.error(f"Failed to set {value} for '{key}' stat")
