import logging
import sys

def setup_logging():
    """
    Setup the logging configuration.
    """

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")

    logging_handler_out = logging.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(logging.INFO)
    logging_handler_out.setFormatter(formatter)
    root.addHandler(logging_handler_out)

    logging_handler_err = logging.StreamHandler(sys.stderr)
    logging_handler_err.setLevel(logging.ERROR)
    logging_handler_err.setFormatter(formatter)
    root.addHandler(logging_handler_err)

    logging.info("Logging setup complete.")
