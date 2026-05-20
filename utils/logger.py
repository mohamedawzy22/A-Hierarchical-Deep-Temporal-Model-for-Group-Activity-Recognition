import os
import logging
from datetime import datetime


def get_logger(exp_name="exp", log_name="run", logs_dir="logs"):

    # folder: logs/exp_name
    path = os.path.join(logs_dir, exp_name)
    os.makedirs(path, exist_ok=True)

    # file name
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(path, f"{log_name}_{time}.log")

    # logger
    logger = logging.getLogger(file_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(message)s")

    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Log: {file_path}")

    return logger