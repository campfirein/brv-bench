"""Common Utilities.

Common Support functions
"""

__author__ = "Danh Doan"
__email__ = "danhdoancv@gmail.com"
__date__ = "2020/04/19"
__status__ = "development"


# =============================================================================


import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# =============================================================================


DEBUG_LENGTH = 79
DEBUG_SEPERATOR = "="


# =============================================================================


def dbg(*args: object) -> None:
    """Debug by print values.

    Args:
    ----
    args (obj) : packed list of values to debug

    """
    caller = inspect.stack()[1]
    module = inspect.getmodule(caller[0])
    caller_module = module.__name__ if module else None
    caller_function, caller_line_number = caller.function, caller.lineno
    dbg_prefix = f"[{caller_module}-{caller_function}:{caller_line_number}]"
    logging.debug(dbg_prefix.ljust(DEBUG_LENGTH, DEBUG_SEPERATOR))

    for arg in args:
        if isinstance(arg, list):
            print_list(arg)
        else:
            logging.debug(arg)


def print_list(lst: list[Any]) -> None:
    """Print items of a list.

    Args:
    ----
    lst (List[obj]) : list of object to print

    """
    for i, item in enumerate(lst):
        logging.debug("%d %s", i + 1, item)


# =============================================================================
