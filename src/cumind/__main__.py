"""Entry point for the cumind package."""

from cumind.utils.logger import log as logger


def main() -> None:
    """Main function for the cumind package."""
    logger.info("Welcome to CuMind!")
    logger.info("This is the main entry point for the CuMind package.")
    logger.info("You can run training scripts or interact with the package from here.")

    logger.close()


if __name__ == "__main__":
    main()
