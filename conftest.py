"""Keep parallel workers from fighting each other over the same cores.

torch sizes its intra-op thread pool to the whole machine. Under ``pytest -n``
that happens once per worker, so a 24-core box runs 24 workers of 24 threads
and spends its time on context switches: the suite measured 15.8s serial and
102s across ``-n auto`` before this, which is the opposite of the point.

One thread per worker, since the work is already parallel at the test level.
Serial runs are left alone -- there is nothing to contend with, and a single
test of a real model should still use the machine.
"""

import os

import torch as th


def pytest_configure(config):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        th.set_num_threads(1)
