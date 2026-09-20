import os
import socket

import pytest


@pytest.fixture(autouse=True)
def _isolate_env():
    saved = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved)


class NetworkBlockedError(RuntimeError):
    """Raised when a test attempts a real network connection."""


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Every test runs offline, as a guarantee rather than a convention.

    URL-fetching tests must route through the sources._client seam with
    httpx.MockTransport. Without this fixture, that only holds because
    every test remembers to mock; the first one that forgets would make a
    real request. Here, it fails loudly instead.

    Set DOCEVAL_LIVE_API=1 to exempt a deliberately live test from the
    block (see the live API test gated on it).
    """
    if os.environ.get("DOCEVAL_LIVE_API") == "1":
        yield
        return

    test_id = request.node.nodeid

    def _blocked(*args, **kwargs):
        raise NetworkBlockedError(
            f"{test_id} attempted a real network connection. Route it "
            "through the sources._client seam with httpx.MockTransport, "
            "or set DOCEVAL_LIVE_API=1 for a deliberately live test."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    yield
