import logging
import os
import time
from datetime import datetime


class A3CLogger:
    """Small training logger used by the refactored A3C code."""

    def __init__(self, config):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("logs", f"A3C_{timestamp}")
        self.log_path = os.path.join(self.log_dir, "train.log")
        self.model_dir = config["logging"]["model_dir"]
        self.start_time = time.time()

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = self._build_logger()

        self.info(f"Logging to {self.log_dir}")
        print(f"Logging to {self.log_dir}")

    def _build_logger(self):
        logger = logging.getLogger("a3c")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.log_path)

        for handler in (console_handler, file_handler):
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

        return logger

    def info(self, message):
        self.logger.info(message)

    def close(self):
        elapsed = int(time.time() - self.start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.info(f"Training completed in {hours}h {minutes}m {seconds}s")
