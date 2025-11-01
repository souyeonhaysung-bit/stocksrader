"""Daily scheduler for Momentum Radar pipeline."""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from run_pipeline import run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def execute_pipeline() -> None:
    try:
        logger.info("Starting pipeline run")
        run_pipeline()
        logger.info("Pipeline run completed successfully")
    except Exception as exc:
        logger.exception("Pipeline run failed: %s", exc)


def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(execute_pipeline, "interval", hours=24, next_run_time=datetime.utcnow())

    logger.info("Scheduler initialized. Running every 24 hours.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shut down.")


if __name__ == "__main__":
    main()
