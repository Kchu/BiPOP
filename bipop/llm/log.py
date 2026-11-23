"""Verbose console output of the planning pipeline, enabled with ``BIPOP_VERBOSE=1``."""

import os

VERBOSE = os.environ.get("BIPOP_VERBOSE", "0") == "1"


def debug_print(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)
