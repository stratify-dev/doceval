import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env():
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)
