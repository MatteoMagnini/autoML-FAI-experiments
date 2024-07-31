import hashlib
import logging
from typing import Iterable


__all__ = [
    "logger",
    "enable_logging",
    "disable_logging",
    "enable_file_logging",
    "disable_file_logging",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARNING",
    "LOG_ERROR",
    "LOG_CRITICAL",
    "LOG_FATAL",
]

logger = logging.getLogger("fai")
logger.setLevel(logging.DEBUG)
logger.propagate = False

LOG_DEBUG = logging.DEBUG
LOG_INFO = logging.INFO
LOG_WARNING = logging.WARNING
LOG_ERROR = logging.ERROR
LOG_CRITICAL = logging.CRITICAL
LOG_FATAL = logging.FATAL
INDENT = ' ' * 2
LOG_FLOAT_PRECISION = 4


def enable_logging(level: int = LOG_INFO):
    """
    Enable logging.
    @param level: the logging level.
    """
    disable_logging()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def disable_logging():
    """
    Disable logging.
    """
    logger.setLevel(logging.CRITICAL)
    logger.handlers = []


def enable_file_logging(filename: str, level: int = LOG_INFO):
    """
    Enable logging to a file.
    @param filename: the filename.
    @param level: the logging level.
    """
    enable_logging(level)
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    logger.addHandler(fh)



def disable_file_logging():
    """
    Disable logging to a file.
    """
    logger.setLevel(logging.CRITICAL)
    if len(logger.handlers) > 1:
        logger.removeHandler(logger.handlers[1])


def exp_name(dataset: str,
             method: str,
             metric: str,
             protected: int,
             lambda_value: float,
             exp_number: int,
             seed: int
             ) -> str:
    """
    Create the experiment file name.

    @param dataset: dataset name
    @param method: method name
    @param protected: protected feature index
    @param metric: fairness metric name
    @param lambda_value: lambda value
    @param exp_number: experiment number
    @param seed: seed
    @return: experiment file name
    """
    string_name = f"{dataset}_protected{protected}_method{method}_fairness{metric}_lambda{lambda_value}_exp{exp_number}_seed{seed}"
    hash_name = hashlib.md5(string_name.encode()).hexdigest()
    return hash_name + ".yml"


def log_experiment_setup(dataset: str,
                         method: str,
                         protected: Iterable[int],
                         metric: str,
                         lambda_value: float,
                         exp_number: int,
                         seed: int
                         ) -> None:
    """
    Log the experiment setup.

    @param dataset: dataset name
    @param method: method name
    @param protected: protected features indices
    @param metric: fairness metric name
    @param lambda_value: lambda value
    @param exp_number: experiment number
    @param seed: seed
    """
    logger.info(f"setup:")
    logger.info(f"{INDENT}dataset: {dataset}")
    logger.info(f"{INDENT}method: {method}")
    logger.info(f"{INDENT}metric: {metric}")
    logger.info(f"{INDENT}protected: {protected}")
    logger.info(f"{INDENT}lambda_value: {lambda_value}")
    logger.info(f"{INDENT}exp_number: {exp_number}")
    logger.info(f"{INDENT}seed: {seed}")
