"""Main Entry Point."""

from utils.logger_utils import setup_logger

# =============================================================================


logger = setup_logger("logs")

# =============================================================================


def app():
    """Application entry point."""
    logger.info("Starting the application")
    logger.info("PROCESSING...")
    logger.info("Application finished")


# =============================================================================


def main():
    """Main entry point."""
    app()


# =============================================================================


if __name__ == "__main__":

    
    main()


# =============================================================================
