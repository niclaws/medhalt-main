"""Shared IST logging and monotonic wall-clock timing (standard library only)."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
import inspect
import logging
from pathlib import Path
import time

LOG = logging.getLogger("rebuttal")
IST = timezone(timedelta(hours=5, minutes=30), "IST")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=IST)
        return (
            dt.strftime(datefmt)
            if datefmt
            else dt.isoformat(sep=" ", timespec="milliseconds") + " IST"
        )


def configure_logging(log_file=None, level="DEBUG"):
    """Configure only our logger; close previous handlers on repeated setup."""
    LOG.setLevel(level.upper())
    LOG.propagate = False
    for handler in LOG.handlers[:]:
        LOG.removeHandler(handler)
        handler.close()
    formatter = ISTFormatter("%(asctime)s | %(levelname)-8s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    LOG.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        LOG.addHandler(handler)
    LOG.info(
        "Logging initialized level=%s timezone=IST (UTC+05:30) file=%s", level, log_file
    )


@contextmanager
def step(label, level=logging.DEBUG):
    """Measure elapsed wall time without synchronizing CUDA or swallowing errors."""
    if not LOG.isEnabledFor(level):
        yield
        return
    start = time.perf_counter()
    LOG.log(level, "START %s", label)
    try:
        yield
    except BaseException as exc:
        LOG.log(
            level,
            "FAILED %s | elapsed=%.3fs | error=%s",
            label,
            time.perf_counter() - start,
            type(exc).__name__,
        )
        raise
    else:
        LOG.log(level, "DONE  %s | elapsed=%.3fs", label, time.perf_counter() - start)


def timed(fn):
    """Time a stage; include only selected scalar identifiers, never prompts/tensors."""
    sig = inspect.signature(fn)
    label = f'{fn.__module__.rsplit(".", 1)[-1]}.{fn.__qualname__}'
    fields = (
        "tag",
        "model_name",
        "source",
        "target",
        "split",
        "variant",
        "rt",
        "layer",
        "path",
        "root",
        "d",
        "name",
        "batch_size",
        "quantization",
    )

    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not LOG.isEnabledFor(logging.DEBUG):
            return fn(*args, **kwargs)
        bound = sig.bind_partial(*args, **kwargs).arguments
        details = [
            f"{k}={bound[k]}"
            for k in fields
            if k in bound and isinstance(bound[k], (str, int, Path))
        ]
        p = bound.get("p")
        if isinstance(p, dict) and "sample_id" in p:
            details.append(f'sample_id={p["sample_id"]}')
        for k in (
            "input_ids",
            "cand_ids",
            "interventions",
            "ivs",
            "examples",
            "samples",
            "specs",
        ):
            if isinstance(bound.get(k), (list, tuple)):
                details.append(f"{k}_n={len(bound[k])}")
        with step(label + (" " + " ".join(details) if details else "")):
            return fn(*args, **kwargs)

    return wrapped


def timed_call(label, fn, /, *args, **kwargs):
    """Time a call into an external library without modifying that library."""
    if LOG.isEnabledFor(logging.DEBUG):
        paths = [str(x) for x in args if isinstance(x, Path)]
        if paths:
            label += " path=" + ",".join(paths)
    with step(label):
        return fn(*args, **kwargs)


def progress(items, label):
    """Log each work item before processing and elapsed time after it completes."""
    if not LOG.isEnabledFor(logging.DEBUG):
        yield from items
        return
    total = len(items) if hasattr(items, "__len__") else "?"
    for i, item in enumerate(items, 1):
        start = time.perf_counter()
        LOG.debug("START %s item=%d/%s", label, i, total)
        yield item
        LOG.debug(
            "DONE  %s item=%d/%s | elapsed=%.3fs",
            label,
            i,
            total,
            time.perf_counter() - start,
        )
