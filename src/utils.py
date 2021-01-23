import time
import contextlib


@contextlib.contextmanager
def timer():
    """ Time the execution of a context block.
    Yields:
        None
    """
    time_start = time.time()

    yield

    elapsed = time.time() - time_start
    print(f'Time: {elapsed:.2f}s')
