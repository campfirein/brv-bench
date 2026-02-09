"""Logger Utility."""

__author__ = "Danh Doan"
__email__ = ("danhdoancv@gmail.com",)
__date__ = "2023/02/28"
__status__ = "development"


# =============================================================================


import logging
import os
import sys
from logging import Handler, Logger
from logging.handlers import TimedRotatingFileHandler

# =============================================================================


FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d] %(message)s"
)


# =============================================================================


def remove_handler_from_logger(logger: Logger, handler_type: type[Handler]) -> None:
    """Remove handler from logger if exist."""
    for handler in logger.handlers:
        if isinstance(handler, handler_type):
            logger.removeHandler(handler)
            handler.close()


# =============================================================================


def setup_logger(
    log_dir: str = "logs",
    use_stream: bool = True,
    use_file: bool = True,
    level: int = logging.DEBUG,
) -> Logger:
    """Create a logger with immediate flushing for localhost."""
    # Create logger with its own namespace
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(level)
    # ------------------------
    # Stream Handler (stdout)
    # ------------------------
    if use_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(FORMATTER)

        # Force flush after each log record
        orig_emit = stream_handler.emit

        def emit(record: logging.LogRecord) -> None:
            """Flush the stream handler immediately after each log record."""
            orig_emit(record)
            stream_handler.flush()

        stream_handler.emit = emit  # type: ignore[method-assign]

        logger.addHandler(stream_handler)

        # Ensure Python stdout line buffering
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    # ------------------------
    # File Handler (optional)
    # ------------------------
    if use_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
