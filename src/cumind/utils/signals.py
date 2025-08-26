import signal
import sys

from cumind.utils.config import cfg
from cumind.utils.logger import log


def register_signals() -> None:
    def handler(signum: int, frame: object) -> None:
        log.info(f"Received signal {signum}, shutting down.")
        log.shutdown()
        if cfg.tracing.enabled:
            from jax.profiler import stop_trace

            stop_trace()  #  type: ignore

        sys.exit(1)

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
