"""Timer Utility."""

__author__ = "Danh Doan"
__email__ = ("danhdoancv@gmail.com",)
__date__ = "2023/02/28"
__status__ = "development"


# =============================================================================


import logging
import time
from collections.abc import Callable
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================


def tiktok(func: Callable) -> Callable:
    """Decorate input function to measure running time.

    Args:
    ----
    func (object) : function to be decorated


    Returns:
    -------
    (object) : function after decorated

    """

    @wraps(func)
    def inner(*args: object, **kwargs: object) -> Callable:
        """Calculate time."""
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_taken_ms = (end - begin) * 1000
        logger.debug(f"Time taken for {func.__name__}: {time_taken_ms:.5f} ms")
        return result

    return inner


# =============================================================================
