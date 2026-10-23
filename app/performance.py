from contextlib import contextmanager
import time


@contextmanager
def timed_operation(label):
    start = start_timer()
    outcome = {"succeeded": None}
    try:
        yield outcome
    except BaseException:
        succeeded = False
        raise
    else:
        succeeded = outcome["succeeded"]
        if succeeded is None:
            succeeded = True
    finally:
        finish_timer(
            label,
            start,
            succeeded,
            outcome.get("details"),
        )


def start_timer():
    return time.perf_counter()


def finish_timer(label, start, succeeded=True, details=None):
    elapsed_ms = (time.perf_counter() - start) * 1000
    status = "success" if succeeded else "failure"
    detail_text = f"; {details}" if details else ""
    print(f"[TIME] {label}: {elapsed_ms:.2f} ms ({status}{detail_text})")
