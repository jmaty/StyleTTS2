import os
import time

from accelerate import Accelerator

from logger import setup_logging, get_logger


def main():
    acc = Accelerator()

    log_dir = os.path.join("Exps", "_logging_test")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    # Configure logging via our helper (console + file only on main process)
    setup_logging("INFO", log_file, accelerator=acc)
    logger = get_logger(__name__)

    # Emit a couple of lines; only main process should print to console and write the file
    logger.info(
        f"hello from rank={getattr(acc, 'process_index', -1)} main={getattr(acc, 'is_main_process', False)}"
    )
    time.sleep(0.2)
    logger.info("second message")

    acc.wait_for_everyone()


if __name__ == "__main__":
    main()

