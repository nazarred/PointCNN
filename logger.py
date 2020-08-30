import logging


def setup_logging(file_path, level=logging.INFO):  # pragma: nocover
    """Set up logging."""
    logger = logging.getLogger()
    logger.setLevel(level)

    syslog_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    syslog_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)
    logger.addHandler(file_handler)

    # ES request logger
    # requests

