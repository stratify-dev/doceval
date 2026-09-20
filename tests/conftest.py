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

    The block is lifted only for a test marked @pytest.mark.live, never by
    ambient environment. DOCEVAL_LIVE_API decides whether a live test is
    allowed to run at all (Task 10's live API test is skipif-gated on it),
    but that is a separate concern from this fixture: a stray exported
    DOCEVAL_LIVE_API=1 must not, by itself, silently disable network
    enforcement for the other 70-odd tests in the suite. A test reaches
    the network only by explicitly declaring that it does, via the marker.
    """
    if request.node.get_closest_marker("live") is not None:
        yield
        return

    test_id = request.node.nodeid

    def _blocked(*args, **kwargs):
        raise NetworkBlockedError(
            f"{test_id} attempted a real network connection. Route it "
            "through the sources._client seam with httpx.MockTransport, "
            "or mark the test @pytest.mark.live if it must be live."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield
